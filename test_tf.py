import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

def test_tf():
    print(tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
    return tf.__version__

def num_zeros(a):
    #takes a tensor (or array) and returns the number of zeros/less-than-zeros in it, used for slicing
    count = 0
    for i in range(len(a)):
        if a[i] <= 0:
            count += 1
    return count

def num_greater_than(a,THRESH):
    #takes a tensor (or array) and returns the number of values greater than given threshold in it, used for slicing (rMax)
    count = 0
    for i in range(len(a)):
        if a[i] >= THRESH:
            count += 1
    return count
if tf.config.list_physical_devices('GPU'):
    device = '/GPU:0'
elif tf.config.list_physical_devices('MPS'):  # For Apple Silicon (M1/M2) GPUs
    device = '/MPS:0'
else:
    device = '/CPU:0'

print(f"Using device: {device}")

test_tf()

aa = tf.constant([1,2,3])
print(aa**2)

bb = tf.constant(.1)
print(bb)

# cum_dist = tf.math.cumsum(tf.convert_to_tensor([[.1,.4,.2],[.4,.3,.3],[.5,.3,.5]]))
# cum_dist /= cum_dist[-1]  # to account for floating point errors
# cum_dist = tf.transpose(cum_dist)
# unif_samp = tf.random.uniform((3,20), 0, 1)
# print(len(unif_samp))
# print(len(cum_dist))
# idxs = tf.searchsorted(cum_dist, unif_samp)
# print(idxs)
# samp = tf.gather(aa, idxs)
# print(samp)
# print(cum_dist)
# print("hi")





#single step bulk and bound return the coefficient by which to multiply dx for change in x and l in simulation (return +/- 1 or 0)
def single_step_bulk(delt,lam,bet,NUM_SAMPLES=1):
  #print(delt,lam,bet)
  #choose = np.random.choice([-1,0,1],p=[delt/(delt+lam+bet),bet/(delt+lam+bet),lam/(delt+lam+bet)])
    choose = tf.convert_to_tensor([-1.0,0.0,1.0])
    probs = tf.convert_to_tensor([delt/(delt+lam+bet),bet/(delt+lam+bet),lam/(delt+lam+bet)])
    # print(probs)
    
    cum_dist = tf.math.cumsum(probs)
    cum_dist /= cum_dist[-1]  # to account for floating point errors
    cum_dist = tf.transpose(cum_dist)
    unif_samp = tf.random.uniform((NUM_SAMPLES,1), 0, 1)
    # print(len(unif_samp))
    # print(len(cum_dist))
    # print(cum_dist)
    idxs = tf.searchsorted(cum_dist, unif_samp)
    samp = tf.gather(choose, idxs)  # samp contains the k weighted samples
    samp = tf.reshape(samp,[-1])

    xDisp = samp
    lDisp = abs(samp) - 1
    rDisp = -1 * lDisp
    return xDisp, lDisp, rDisp

def single_step_bound(gam,lam,bet,NUM_SAMPLES=1):
    choose = tf.convert_to_tensor([-1.0,0.0,1.0])
    probs = tf.convert_to_tensor([gam/(gam+lam+bet),lam/(gam+lam+bet),bet/(gam+lam+bet)])
    # choose = np.random.choice([-1,0,1], p=[gam/(gam+lam+bet),bet/(gam+lam+bet),lam/(gam+lam+bet)])

    cum_dist = tf.math.cumsum(probs)
    cum_dist /= cum_dist[-1]  # to account for floating point errors
    cum_dist = tf.transpose(cum_dist)
    unif_samp = tf.random.uniform((NUM_SAMPLES,1), 0, 1)
    # print(len(unif_samp))
    # print(cum_dist)
    idxs = tf.searchsorted(cum_dist, unif_samp)
    samp = tf.gather(choose, idxs)  # samp contains the k (NUM_SAMPLES) weighted samples
    samp = tf.reshape(samp,[-1])

    rDisp = samp
    xDisp = abs(abs(samp) - 1)
    lDisp = -1*rDisp
    # if(choose == 0):
    #     lDisp = -1
    #     xDisp = 0
    # elif(choose == -1):
    #     lDisp = 1
    #     xDisp = 0
    # else:
    #     lDisp = 0
    #     xDisp = 1
    # rDisp = -1 * lDisp
    return xDisp, lDisp, rDisp

#combine for a single trial
def single_trial(x0,dx,R,b,delt,lam,bet,gam,rMax=100,l0=0,output=False):
    #x0, l0 should be tensors
    print(x0)
    currX = x0
    currL = l0
    currR = R - currL
    counters = tf.zeros(len(x0))
    numComplete = 0
    numAch = 0
    numHitRMax = 0

    #each of the rate functions should operate on and return tensors
    currDelta = delt(currX,currL,currR,dx)
    currLambda = lam(currX,currL,currR,dx)
    currBeta = bet(currX,currL,currR,dx)
    currGamma = gam(currX,currL,currR,dx)

    # if(currR <= 0):
    #   reachedR = 1
    #   print("R is already reached")
    #   return reachedR

    while(numComplete < len(x0)):
        rInds = tf.argsort(currR)
        currX = tf.gather(currX,rInds)
        currL = tf.gather(currL,rInds)
        currR = tf.gather(currR,rInds)
        counters = tf.gather(counters,rInds)
        # print(currX)
        # print(currR)

        rZeros = num_zeros(currR)
        # print("rZeros: " + str(rZeros))
        rMaxNum = num_greater_than(currR,rMax)
        rAch,currR,rHitMax = tf.split(currR,[rZeros,len(currR)-rZeros-rMaxNum,rMaxNum],0)
        xAch,currX,xHitRMax = tf.split(currX,[rZeros,len(currX)-rZeros-rMaxNum,rMaxNum],0)
        lAch,currL,lHitRMax = tf.split(currL,[rZeros,len(currL)-rZeros-rMaxNum,rMaxNum],0)
        counterAch,counters,counterHitRMax = tf.split(counters,[rZeros,len(counters)-rZeros-rMaxNum,rMaxNum],0)
        #these are the values which will be returned at the end of the iteration
        numAch = len(rAch)
        numHitRMax = len(rHitMax)
        numComplete = numAch + numHitRMax

        #at this point, we have split the x, l, and r into three groups: those where r reached 0, those that are still viable, and those that have hit rMax
        #now we perform our operations on the viable arrays for one iteration and then join them back together and evaluate numComplete

        #first split into currX > 0 and currX <= 0
        xInds = tf.argsort(currX)
        currX = tf.gather(currX,xInds)
        currL = tf.gather(currL,xInds)
        currR = tf.gather(currR,xInds)
        counters = tf.gather(counters,xInds)

        
        # print("xZeros: " + str(xZeros))
        # print(currX)
        # print(counters)
        # print([xZeros,len(currX)-xZeros])
        xZeros = num_zeros(currX)
        xLess,xGreater = tf.split(currX,[num_zeros(currX),len(currX)-num_zeros(currX)],0)
        lLess,lGreater = tf.split(currL,[num_zeros(currX),len(currL)-num_zeros(currX)],0)
        rLess,rGreater = tf.split(currR,[num_zeros(currX),len(currR)-num_zeros(currX)],0)
        counterLess,counterGreater = tf.split(counters,[xZeros,len(counters)-xZeros],0)

        xLess = tf.zeros(len(xLess))

        #reevaluating rates
        currDeltaLess = delt(xLess,lLess,rLess,dx)
        currLambdaLess = lam(xLess,lLess,rLess,dx)
        currBetaLess = bet(xLess,lLess,rLess,dx)
        currGammaLess = gam(xLess,lLess,rLess,dx)

        currDeltaGreater = delt(xGreater,lGreater,rGreater,dx)
        currLambdaGreater = lam(xGreater,lGreater,rGreater,dx)
        currBetaGreater = bet(xGreater,lGreater,rGreater,dx)
        currGammaGreater = gam(xGreater,lGreater,rGreater,dx)

        #taking a step!
        # print(len(xLess))
        if(len(xLess) > 0):
            xDispLess,lDispLess,rDispLess = single_step_bound(currGammaLess,currLambdaLess,currBetaLess,NUM_SAMPLES=len(xLess))
        else:
            xDispLess,lDispLess,rDispLess = tf.zeros(0),tf.zeros(0),tf.zeros(0)
        if(len(xGreater) > 0):
            xDispGreater,lDispGreater,rDispGreater = single_step_bulk(currDeltaGreater,currLambdaGreater,currBetaGreater,NUM_SAMPLES=len(xGreater))
        else:
            xDispGreater,lDispGreater,rDispGreater = tf.zeros(0),tf.zeros(0),tf.zeros(0)

        currX = tf.concat([xLess + xDispLess*dx, xGreater + xDispGreater*dx],0)
        currL = tf.concat([lLess + lDispLess*dx, lGreater + lDispGreater*dx],0)
        currR = R - currL
        counters = tf.concat([counterLess + 1, counterGreater + 1],0)

        #combining everything again
        currX = tf.concat([currX,xAch,xHitRMax],0) 
        currL = tf.concat([currL,lAch,lHitRMax],0)
        currR = tf.concat([currR,rAch,rHitMax],0)
        counters = tf.concat([counters,counterAch,counterHitRMax],0)

        currX = tf.convert_to_tensor(np.round(currX.numpy(),1))
        currL = tf.convert_to_tensor(np.round(currL.numpy(),1))
        currR = tf.convert_to_tensor(np.round(currR.numpy(),1))
    
    return numAch, numHitRMax, numComplete

"""
def single_trial_graph(x0,dx,R,b,delt,lam,bet,gam,epsilon,rMax = 100,l0=0,output=False):

    rList = []
    xList = []


    currX = x0
    currL = l0
    currR = R - currL

    currDelta = delt(currX,currL,currR,dx,epsilon)
    currLambda = lam(currX,currL,currR,dx,epsilon)
    currBeta = bet(currX,currL,currR,dx,epsilon)
    currGamma = gam(currX,currL,currR,dx,epsilon)

    reachedR = -1

    # if(currR <= 0):
    #   reachedR = 1
    #   print("R is already reached")
    #   return reachedR

    while_counter = 0
    while(currR < rMax and (currL < R or currX > 0)):
        rList.append(currR)
        xList.append(currX)
        while_counter += 1
        #reevaluating rates based on whatever functions they are (defined I'm assuming in terms of some combination of x, l, and r)
        currDelta = delt(currX,currL,currR,dx,epsilon)
        currLambda = lam(currX,currL,currR,dx,epsilon)
        currBeta = bet(currX,currL,currR,dx,epsilon)
        currGamma = gam(currX,currL,currR,dx,epsilon)
        #taking a step!
        if(currX > 0):
            xDisp,lDisp,rDisp = single_step_bulk(currDelta,currLambda,currBeta)
            currX += xDisp * dx
            currL += lDisp * dx
            currR = R - currL
            currX = round(currX,4)
            currL = round(currL,4)
            currR = round(currR,4)
        elif(currX <= 0):
            currX = 0
            currGamma = gam(currX,currL,currR,dx,epsilon)
            currLambda = lam(currX,currL,currR,dx,epsilon)
            currBeta = bet(currX,currL,currR,dx,epsilon)
            xDisp,lDisp,rDisp = single_step_bound(currGamma,currLambda,currBeta)
            currX += xDisp * dx
            currL += lDisp * dx
            currR = R - currL
            currX = round(currX,4)
            currL = round(currL,4)
            currR = round(currR,4)
        else:
            if(output):
                print("edge case: exiting after " + str(while_counter) + " iterations")
            # if(output):
            #   print("currR = " + str(currR))

    if(output):
        print("success! exiting after " + str(while_counter) + " iterations")

    rList.append(currR)
    xList.append(currX)

    if(currR >= rMax):
        reachedR = 0
        print("max R is reached: " + str(rMax))
    elif(currL >= R or currR <= 0):
        reachedR = 1
        print("R is reached (resource depleted)")
    #   elif(while_counter >= 2000000):
    #     reachedR = 0
    #     print("max iterations reached")

    return rList, xList, reachedR
"""

#monte carlo
def get_prob_number(num_trials,x0,dx,R,b,delt,lam,bet,gam,epsilon,rMaxx=100,l00=0,outputt=False):
    total = 0
    for i in range(num_trials):
        total += single_trial(x0,dx,R,b,delt,lam,bet,gam,rMax=rMaxx,l0=l00,output=outputt)
    return total/float(num_trials)


#THIS AND BELOW NEEDS TO BE FIXED
"""
"""
"""
"""
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#trying beta = gamma = 1/(ln(r+1)) THIS IS A NONSENSE COMMENT
num_trials = 10
BIGR = 2
trialX = np.linspace(0.1,2,5)
trialR = np.linspace(0.1,2,5) # making the dimensions for my own orientation purposes
alph = 1.5
alphs = [.1*i for i in range(1,36)]
# PROBS = np.empty(np.array(alphs).shape)
# epsilon = 1

NUMS_ACH = np.empty(np.array(alphs).shape)
NUMS_HIT_R_MAX = np.empty(np.array(alphs).shape)

# X, R = np.meshgrid(trialX, trialR)
# PROBS = np.empty((5,5))
#dims are PROBS[r][x]

def lambdaa(x,l,r,dx):
    return r

def deltaa(x,l,r,dx):
    return r

def gammaa(x,l,r,dx):
    return r**alph

def betaa(x,l,r,dx):
    return dx*r

# for r in range(len(PROBS)):
#   for x in range(len(PROBS[r])):
#     print("r: "+str(trialR[r])+"  x: " + str(trialX[x]))
#     PROBS[r][x]=get_prob_number(num_trials,trialX[x],.1,BIGR,50,deltaa,lambdaa,betaa,gammaa,epsilon,rMaxx=2000,l00 = BIGR-trialR[r],outputt=False)
with tf.device(device):
    for a in range(len(alphs)):
        def gammaa(x,l,r,dx):
            return r**alphs[a]
        print("alpha: " + str(alphs[a]))
        xxxx = time.time()
        print("time: " + str(xxxx))
        NUMS_ACH[a], NUMS_HIT_R_MAX[a], _ = single_trial(tf.ones(num_trials),.1,BIGR,50,deltaa,lambdaa,betaa,gammaa,rMax=2000,l0 = BIGR-tf.ones(num_trials),output=False)
        print("time taken: " + str(time.time()-xxxx))
        #get_prob_number(num_trials,1,.1,BIGR,50,deltaa,lambdaa,betaa,gammaa,epsilon,rMaxx=2000,l00 = BIGR-1,outputt=True)

    print(NUMS_ACH)
    print(NUMS_HIT_R_MAX)
    PROBS = [NUMS_ACH[a]/num_trials for a in range(len(alphs))]

    with open('/Users/Greencat/sustainability-simulations/diff_alphas/diffAlphsStarting11_maxR2000_tfm1.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows([PROBS])