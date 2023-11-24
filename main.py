import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from numba import prange, njit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os, copy, random, collections, argparse
from pathlib import Path
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

class FeatureBlock(nn.Module):
    def __init__(self, shapelet, len_ts, iShapelet, pDic={}):
        super(FeatureBlock, self).__init__()
        self.len_ts = len_ts
        L = shapelet.shape[-1]
        self.shapelet_info = pDic["shapelets_info"][iShapelet]
        i,m,j = self.shapelet_info["pos"]
        self.m_ts = m
        self.dilation = self.shapelet_info["dilation"]
        self.scaleType = self.shapelet_info["scaleType"]
        tmp = torch.FloatTensor(shapelet)
        self.shapelet = nn.Parameter(tmp.view(L), requires_grad=True)
        self.posMap = nn.Parameter(torch.zeros(pDic["Q"] - (self.shapelet.shape[-1]-1)*self.dilation), requires_grad=True)
        self.posSlope = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.pDic = pDic
        self.iShapelet = iShapelet

    def gen_unfold(self, x):
        info = self.shapelet_info
        m, L, dilation, scaleType  = info["pos"][1], info["L"], info["dilation"], info["scaleType"]
        eps = self.pDic["scale_epsilon"]
        x = x[:,m:m+1,:]
        mean, std = None, None
        x_unfold = x.unfold(2, (L-1)*dilation+1, 1)[:,:,:,::dilation].contiguous()
        if scaleType==2:
            std = torch.std(x_unfold, dim=3).unsqueeze(3)
            mean = torch.mean(x_unfold, dim=3).unsqueeze(3)
            x_unfold = (x_unfold - mean) / (std + eps)
        elif scaleType==1:
            mean = torch.mean(x_unfold, dim=3).unsqueeze(3)
            x_unfold = x_unfold - mean
        return x_unfold, mean, std

    def forward(self, x, ep):
        x_unfold, mean, std = self.gen_unfold(x)
        del x
        if ep < self.pDic["sgeS"]:
            d_segs = (self.shapelet.detach().view(1,1,1,-1) - x_unfold)**2
        else:
            d_segs = (self.shapelet.view(1,1,1,-1) - x_unfold)**2
        del x_unfold
        ftrsPart0 = d_segs.mean(axis=3)
        self.raw_ftrs = ftrsPart0
        del d_segs
        self.ftrs = (self.raw_ftrs * ((self.posSlope*torch.nn.functional.elu(-self.posMap)+1.0)+1.0).view(1,1,-1)).unsqueeze(3)
        x = torch.sum(self.ftrs, dim=1, keepdim=True).transpose(2, 3)
        x = torch.squeeze(x,2)
        self.x_agg, self.x_argAgg = torch.min(x, 2)
        self.x_agg = torch.relu(self.x_agg)
        return self.x_agg

class ShapeletLayer(nn.Module):
    def __init__(self, shapelets , len_ts, pDic):
        super(ShapeletLayer, self).__init__()
        self.blocks = nn.ModuleList([
            FeatureBlock(shapelet=shapelets[i], len_ts=len_ts, iShapelet=i, pDic=pDic)
            for i in range(len(shapelets))])

    def forward(self, x, ep):
        out = torch.FloatTensor([]).to(x.device)
        for block in self.blocks:
            out = torch.cat((out, block(x,ep=ep)), dim=1)
        return out.view(out.size(0),1,out.size(1))

class LearningShapeletsModel(nn.Module):
    def __init__(self, time_series, targets, pDic={}, show_progress=False):
        super(LearningShapeletsModel, self).__init__()

        self.pDic = pDic
        pDic["I"], pDic["M"], pDic["Q"] = time_series.shape
        pDic["nClass"] = np.unique(targets.detach().cpu().numpy()).shape[0]
        pDic.setdefault('maxItr', 1000)
        pDic.setdefault('maxEpoch', pDic["maxItr"])
        pDic.setdefault('weight_decay', 0.01)
        pDic.setdefault('lr', 0.01)
        pDic.setdefault('dropoutPreY', 0.1)
        pDic.setdefault('bCurriculumDropout', False)
        pDic.setdefault('maxFSOuterItr', 10)
        pDic.setdefault('maxFSInnerItr', 1000)
        pDic.setdefault('maxMemSegments', np.Inf)
        pDic.setdefault('fs_score_epsilon', 1e-5)
        pDic.setdefault('Lcands', [min(11,pDic["Q"])])
        pDic.setdefault('dilationCands', None)
        pDic.setdefault('batch_size', None)
        pDic.setdefault('sgeFtr', 1)
        pDic.setdefault('sgeS', 1)
        pDic.setdefault('geRegS', pDic["sgeS"])
        pDic.setdefault('similarSegPercentile', 25)
        pDic.setdefault('numFSK', 20)
        pDic.setdefault('fs_solver', "lbfgs")
        pDic.setdefault('fs_max_iter', 100)
        pDic.setdefault('scale_types', (2,0,8))
        pDic.setdefault('scale_epsilon', 1e-8)
        pDic.setdefault("randseed", None)
        pDic.setdefault('bPrint', False)
        pDic.setdefault('pos_epsilon', 0.0001)
        pDic.setdefault('lamConti1', 0.0)
        pDic.setdefault('lamConti2', 0.00001)
        self.bFirstRegS = True
        self.memSegments = {}
        shapelets, shapelets_info = self.discoverShapelets(time_series,targets,pDic=pDic,show_progress=show_progress)
        pDic["shapelets_info"] = shapelets_info
        pDic["init_shapelets"] = shapelets.copy()
        self.pDic["K"] = len(shapelets)
        self.shapelet_layer = ShapeletLayer(shapelets=shapelets, len_ts=pDic["Q"], pDic=pDic)
        self.linear3 = nn.Linear(self.pDic["K"], pDic["nClass"], bias=True)

    def forward(self, x, ep, curProgress=1, maxProgress=1):
        y = self.shapelet_layer(x,ep)
        y = torch.relu(y)
        if self.pDic["bCurriculumDropout"]:
            gamma = 10.0 / maxProgress
            dropoutPreY = 1. - ((1.-self.pDic["dropoutPreY"]) * np.exp(-gamma*curProgress) + self.pDic["dropoutPreY"])
            y = F.dropout(y, dropoutPreY, training=self.training)
        else:
            y = F.dropout(y, self.pDic["dropoutPreY"], training=self.training)
        if ep < self.pDic["sgeFtr"]:
            y = self.linear3(y.detach())
        else:
            y = self.linear3(y)
        y = torch.squeeze(y, 1)
        return y

    def predict_obj(self, x, true_y, ep, TX, TY, curProgress, maxProgress):
        y = self(x, ep, curProgress=curProgress, maxProgress=maxProgress)
        regConti = torch.tensor(0., requires_grad=True)
        for block in self.shapelet_layer.blocks:
            tmp = block.posMap[:]
            regConti = regConti + self.pDic["lamConti2"] * torch.sum((tmp[1:]-tmp[:-1])**2) + self.pDic["lamConti1"] * torch.sum(torch.abs(tmp[1:]-tmp[:-1]))
        losses = basic_smooth_crossentropy(y, true_y)
        loss = losses.mean()
        obj = loss + regConti
        return obj

    @staticmethod
    @njit(fastmath=True, cache=True, nogil=False)
    def _genSegments(x, L, eps, dilation, scale_type):
        assert( len(x.shape) == 2 )
        I, Q = x.shape
        J = Q - (L-1)*dilation
        S = np.empty((I,J,L))
        means = np.empty((I,J))
        stds = np.empty((I,J))
        for i in prange(I):
            for j in prange(J):
                seg = x[i,j:j+(L-1)*dilation+1][::dilation]
                means[i,j] = seg.mean()
                stds[i,j] = seg.std() + eps
                if scale_type==2:
                    seg = (seg - means[i,j]) / stds[i,j]
                elif scale_type==1:
                    seg = ( seg - means[i,j] )
                elif scale_type==0:
                    pass
                S[i,j,:] = seg
        return S,means,stds
    
    def genSegments(self, x, L, dilation=1, scale_type=2, m=0, bUseMem=False):
        if bUseMem is False:
            return self._genSegments(x, L, eps=self.pDic["scale_epsilon"], dilation=dilation, scale_type=scale_type)
        if (L,dilation,scale_type,m) in self.memSegments.keys():
            return self.memSegments[(L,dilation,scale_type,m)]
        else:
            result = self._genSegments(x, L, eps=self.pDic["scale_epsilon"], dilation=dilation, scale_type=scale_type)
            if self.sizeMemSegments < self.pDic["maxMemSegments"]:
                self.memSegments[(L,dilation,scale_type,m)] = result
                self.sizeMemSegments += np.prod(result[0].shape)
            return result
            
    def initMemSegments(self):
        del self.memSegments
        self.memSegments = {}
        self.sizeMemSegments = 0
    
    @staticmethod
    @njit(fastmath=True, cache=True, nogil=False)
    def isNovelSeg(seg, cands, th, j, jcands):
        I_,L = cands.shape
        th *= L
        for i in prange(I_):
            tmp = 0.0
            for l in range(L):
                tmp += (seg[l] - cands[i,l])**2
            if tmp <= th and (jcands[i,0] <= j <= jcands[i,1]):
                return False
        return True
    
    @staticmethod
    @njit(fastmath=True, cache=True, nogil=False)
    def seg2dst(T_all, s):
        I,J,L = T_all.shape
        result = np.empty((I,J))
        for i in prange(I):
            for j,v in enumerate(T_all[i]):
                tmp = 0.0
                for l in range(L):
                    tmp += (v[l]-s[l])**2
                result[i,j] = tmp
        result /= L
        return result
    
    @staticmethod
    @njit(fastmath=True, cache=True, nogil=False)
    def dst2ftr(distMat):
        I, J = distMat.shape
        result = np.empty((I,1))
        for i in prange(I):
            result[i,0] = np.min(distMat[i,:])
        return result
    
    def getFtrs(self, X, S, bScale=True, shapelets_info=None):
        shapelets_info = self.pDic["shapelets_info"] if shapelets_info is None else shapelets_info
        I, ch, Q = X.shape
        fX = np.empty([I, 0])
        for k,info in enumerate(shapelets_info):
            s = S[k]
            dilation, scaleType = info["dilation"], info["scaleType"]
            _,m,_ = info["pos"]
            T_all,org_means,org_stds = self.genSegments(X[:,m,:], s.shape[-1], dilation=dilation, scale_type=scaleType)
            tmp = self.dst2ftr(self.seg2dst(T_all[:,info["start"]:info["end"]+1,:], s))
            fX = np.hstack([fX, tmp[:,:]])
        if bScale:
            return self.scalerFS.transform(fX)
        else:
            return fX

    def discoverShapelets(self, X,Y, pDic={}, show_progress=True):
        simplefilter("ignore", category=ConvergenceWarning)
        self.initNovelCounts = []
        self.initNumAllCount = 0
        if type(X)!=np.ndarray:
            X = X.to('cpu').detach().numpy().copy().astype(float)
        if type(Y)!=np.ndarray:
            Y = Y.to('cpu').detach().numpy().copy().astype(int)
        nClass = len(set(Y))
        assert(nClass >=2)
        I, ch, Q = X.shape
        self.initMemSegments()
        thSimilarSegDic={}
        for L in pDic["Lcands"]:
            for iSt, vSt in enumerate(pDic["scale_types"]):
                if vSt <= 0:
                    continue
                tmp = []
                for n in range(300):
                    m = np.random.randint(0,ch)
                    T_all,org_means,org_stds = self.genSegments(X[:,m,:], L, dilation=1, scale_type=iSt, m=m, bUseMem=True)
                    i = np.random.randint(0,T_all.shape[0])
                    i_ = np.random.randint(0,T_all.shape[0])
                    j = np.random.randint(0,T_all.shape[1])
                    j_ = np.random.randint(0,T_all.shape[1])
                    tmp += [ np.mean((T_all[i,j,:]-T_all[i_,j_,:])**2, axis=0) ]
                thSimilarSegDic[(L,iSt)] = np.percentile(tmp, pDic["similarSegPercentile"], axis=0)
        del tmp
        del T_all
        
        fXmerge = np.empty([I, 0])
        acceptSmerge = []
        pbar = range(pDic["maxFSOuterItr"])
        if show_progress == True:
            from tqdm import tqdm_notebook as tqdm
            pbar = tqdm(pbar)
        for nMerge in pbar:
            n_shapelets = pDic["maxFSInnerItr"]
            lengths = np.random.choice(pDic["Lcands"], size=n_shapelets).astype(np.int64)
            if pDic["dilationCands"] is None:
                upper_bounds = np.log2(np.floor_divide(Q - 1, lengths - 1))
                powers = np.zeros(n_shapelets)
                for i in prange(n_shapelets):
                    powers[i] = np.random.uniform(0, upper_bounds[i])
                dilations = np.floor(np.power(2, powers)).astype(np.int64)
            else:
                dilations = np.random.choice(pDic["dilationCands"], size=n_shapelets).astype(np.int64)
            scaleTypes = np.array(random.choices([0,1,2], weights=pDic["scale_types"], k=n_shapelets))

            candidateSDic = {}
            candidateJDic = {}
            acceptS = [] 
            fX = np.empty([I, 0])
            for n in range(pDic["maxFSInnerItr"]):
                L, dilation, scaleType = lengths[n].item(), dilations[n].item(), scaleTypes[n].item()
                thSimilarSeg = thSimilarSegDic[(L,scaleType)]

                m = np.random.randint(0,ch)
                T_all,org_means,org_stds = self.genSegments(X[:,m,:], L, dilation=dilation, scale_type=scaleType, m=m, bUseMem=True)
                i = np.random.randint(0,T_all.shape[0])
                j = np.random.randint(0,T_all.shape[1])

                if not (m,dilation,scaleType,L) in candidateSDic.keys():
                    candidateSDic[(m,dilation,scaleType,L)] = np.empty([0, L], dtype=np.float64)
                    candidateJDic[(m,dilation,scaleType,L)] = np.empty([0, 2], dtype=np.int64)
                tmp1 = candidateSDic[(m,dilation,scaleType,L)]
                tmp2 = candidateJDic[(m,dilation,scaleType,L)]
                self.initNumAllCount += 1
                if len(tmp1)==0 or self.isNovelSeg(T_all[i,j,:], tmp1, thSimilarSeg, j, tmp2):
                    self.initNovelCounts += [n]
                    seg = T_all[i,j,:].copy()
                    mean = org_means[i,j]
                    std = org_stds[i,j]
                    candidateSDic[(m,dilation,scaleType,L)] = np.vstack([candidateSDic[(m,dilation,scaleType,L)], T_all[np.newaxis,i,j,:]])
                    start, end = 0, Q-(L-1)*dilation
                    newX = self.dst2ftr(self.seg2dst(T_all[:,start:end+1,:], seg))

                    candidateJDic[(m,dilation,scaleType,L)] = np.vstack([candidateJDic[(m,dilation,scaleType,L)], np.array([start,end])])
                    acceptS += [(seg,dilation,scaleType,i,m,j,mean,std,start,end)]
                    fX = np.hstack([fX, newX[:,:]])
            
            fX2 = StandardScaler(with_mean=True).fit_transform(fX)
            nGroupMembers = int(fX.shape[1]/len(acceptS))
            clf1 = LogisticRegressionWithFS(nGroupMembers=nGroupMembers, pDic=pDic, C=1.0, penalty='l1', solver='liblinear').fit(fX2,Y)
            acceptS = [ v for i,v in enumerate(acceptS) if i in clf1.idxs ]
            fX = clf1.reduceFX(fX)
            acceptSmerge += copy.deepcopy(acceptS)
            fXmerge = np.hstack([fXmerge, copy.deepcopy(fX)])
        
        fX2 = StandardScaler(with_mean=True).fit_transform(fXmerge)
        nGroupMembers = int(fXmerge.shape[1]/len(acceptSmerge))
        clf1 = LogisticRegressionWithFS(nGroupMembers=nGroupMembers, pDic=pDic, C=1.0, penalty='l1', solver='liblinear').fit(fX2,Y)
        acceptSmerge = [ v for i,v in enumerate(acceptSmerge) if i in clf1.idxs ]
        fXmerge = clf1.reduceFX(fXmerge)
                
        shapelets_info = []
        shapelets = []
        for v1,dilation,scaleType,i,m,j,mean,std,start,end in acceptSmerge:
            shapelets_info += [{"pos":(i,m,j),"dilation":dilation,"scaleType":scaleType,"mean":mean,"std":std,"start":start,"end":end,"L":len(v1),}]
            shapelets += [v1]

        self.scalerFS = StandardScaler(with_mean=True).fit(fXmerge)
        self.clfFS = LogisticRegression(C=1.0, penalty='l1', solver='liblinear', random_state=pDic["randseed"]).fit(self.scalerFS.transform(fXmerge),Y)

        if pDic["bPrint"]:
            tmp = collections.Counter([ v["end"]-v["start"] for v in shapelets_info ])
            tmp1 = ",".join([ "%d:%d" % (v1,v2) for v1, v2 in sorted(tmp.items(), key=lambda x: x[1])[::-1] ])
            tmp = collections.Counter([ len(v[0]) for v in acceptS ])
            tmp2 = ",".join([ "%d:%d" % (v1,v2) for v1, v2 in sorted(tmp.items(), key=lambda x: x[1])[::-1] ])
            print("acceptS %s interval %s dilation %s %s memMega %.3f" % (tmp2, tmp1, sorted(set([ v["dilation"] for v in shapelets_info ])), set(dilations), self.sizeMemSegments/1000000 ))

        self.initMemSegments()
        return shapelets, shapelets_info

def basic_smooth_crossentropy(pred, target, smoothing=0.1):
    n_class = pred.size(1)
    true_dist = torch.full_like(pred, fill_value=smoothing / (n_class - 1)).to(pred.device)
    true_dist.scatter_(dim=1, index=target.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)
    return torch.mean(torch.sum(-true_dist * log_prob, dim=1))

def train(model, TXtra, TYtra, pDic, device, bPrint=False, show_progress=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=pDic["lr"], weight_decay=pDic["weight_decay"])
    Dtra = torch.utils.data.TensorDataset(TXtra, TYtra)
    if ("batch_size" in pDic.keys()) and (not pDic["batch_size"] is None):
        batch_size = pDic["batch_size"]
    else:
        if len(Dtra) < 100:
            batch_size = 16
        elif len(Dtra) < 200:
            batch_size = 32
        elif len(Dtra) < 400:
            batch_size = 64
        elif len(Dtra) < 800:
            batch_size = 128
        else:
            batch_size = 256
    if bPrint:
        print("batch size %d" % (batch_size))
    tra_loader = torch.utils.data.DataLoader(dataset=Dtra, batch_size=batch_size, shuffle = True)
    
    maxProgress = len(tra_loader)*pDic["maxEpoch"]
    curProgress = 0
    for epoch in range(pDic["maxEpoch"]):
        model.train()
        for batchIdx,batch in enumerate(tra_loader):
            inputs, targets = (b.to(device) for b in batch)
            optimizer.zero_grad()
            curProgress +=1
            loss = model.predict_obj(inputs, targets, ep=epoch, TX=TXtra, TY=TYtra, curProgress=curProgress, maxProgress=maxProgress)
            loss.backward()
            optimizer.step()
    return

class LogisticRegressionWithFS(LogisticRegression):
    def __init__(self, nGroupMembers, pDic, C=1.0, penalty='l1', solver='liblinear'):
        self.nGroupMembers = nGroupMembers
        self.pDic = pDic
        super().__init__(C=C, penalty=penalty, solver=solver)

    def fit(self, fX, Y):
        super().fit(fX, Y)
        magnitude = (self.coef_**2).mean(axis=0).reshape(self.nGroupMembers,-1).sum(axis=0)
        self.idxs = np.argsort(magnitude)[::-1][:self.pDic["numFSK"]]
        self.clf2 = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=self.pDic["randseed"]).fit(self.reduceFX(fX),Y)
        return self

    def predict(self, fX):
        return self.clf2.predict(self.reduceFX(fX))
    
    def reduceFX(self, fX):
        idxs = np.sort(np.hstack([ self.nGroupMembers*self.idxs + i for i in range(self.nGroupMembers) ]))
        return fX[:,idxs]

def normalize_standard(X, scaler=None):
    shape = X.shape
    data_flat = X.flatten()
    if scaler is None:
        scaler = StandardScaler()
        data_transformed = scaler.fit_transform(data_flat.reshape(np.product(shape), 1)).reshape(shape)
    else:
        data_transformed = scaler.transform(data_flat.reshape(np.product(shape), 1)).reshape(shape)
    return data_transformed, scaler

def normalize_data(X, scaler=None):
    if scaler is None:
        X, scaler = normalize_standard(X)
    else:
        X, scaler = normalize_standard(X, scaler)
    return X, scaler

def get_ucr_data(file_path, scaler=None):
    data = np.loadtxt(file_path ,delimiter="\t")
    label = data[:,0].astype('int64')-1
    data = data[:,1:]
    if np.isnan(data).any():
            assert(False)
    data, scaler = normalize_data(data[:,np.newaxis,:])
    return data, label, scaler

def correct_label(Y, corrects=None):
    newY = np.zeros(len(Y), dtype=int)-1
    if corrects is None:
        corrects = []
        for i,y in enumerate(sorted(set(Y))):
            corrects += [y]
    for newLabel, oldLabel in enumerate(corrects):
        idxs = np.array([ i_ for i_,y in enumerate(Y) if y==oldLabel])
        newY[idxs] = newLabel
    assert(all(newY>=0))
    return newY, corrects

def eval_accuracy(model, TX, TY):
    bTrain = model.training
    model.eval()
    with torch.no_grad():
        predictions = model(TX,ep=0)
        correct = torch.argmax(predictions.data, 1) == TY
        acc = 1.0 * correct.cpu().detach().numpy().sum()/len(TY)
    if bTrain:
        model.train()
    return acc

def main(args):
    device = device = torch.device(args.device)
    train_file_path = os.path.join(args.dataset_dir, Path(args.dataset_dir).name+"_TRAIN.tsv")
    test_file_path = os.path.join(args.dataset_dir, Path(args.dataset_dir).name+"_TEST.tsv")
    pDic = {"dropoutPreY":args.dropout, "numFSK":args.K, }

    Xtra, Ytra, scaler = get_ucr_data(train_file_path)
    Xtest, Ytest, _ = get_ucr_data(test_file_path, scaler=scaler)
    Ytra, labelCorrects = correct_label(Ytra)
    Ytest, _ = correct_label(Ytest, labelCorrects)

    TXtra = torch.Tensor(Xtra).to(device)
    TYtra = torch.tensor(Ytra, dtype=torch.int64).to(device)
    TXtest = torch.Tensor(Xtest).to(device)
    TYtest = torch.tensor(Ytest, dtype=torch.int64).to(device)

    model = LearningShapeletsModel(TXtra, TYtra, pDic=pDic,).to(device)
    train(model, TXtra, TYtra, pDic, device)
    result = eval_accuracy(model, TXtest, TYtest)
    print("Test accuracy %f" % (result),)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="C:/hoge/UCRArchive_2018/ItalyPowerDemand", type=str, help="Dataset directory")
    parser.add_argument("--device", default="cpu", type=str, help="Device for training model")
    parser.add_argument("--K", default="30", type=int, help="The number of shapelets")
    parser.add_argument("--dropout", default="0.25", type=float, help="Dropout rate")
    args = parser.parse_args()
    print(args)
    main(args)
    print("END")
