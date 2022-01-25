def jaccard(output,target,eps=1e-5):
  target=target.float()
  intersection=((output*target).sum())+eps
  union=(output.sum()+target.sum()-intersection)+eps
  return 1-(intersection/union)
  
def softmax_jaccard(output, target, eps=1e-5):
    target_0=(target==0).float()
    target_255=(target==255).float()
    output_0=output[:,0,...]
    output_1=output[:,1,...]
    jaccard_0=jaccard(output_0,target_0)
    jaccard_1=jaccard(output_1,target_255)
    return ((jaccard_0+jaccard_1)/2.)
def sigmoid_jaccard(output, target, eps=1e-5):
    target_0=(target==0).float()
    target_255=(target==255).float()
    output_0=1-output
    jaccard_0=jaccard(output_0,target_0)
    jaccard_1=jaccard(output,target_255)
    return ((jaccard_0+jaccard_1)/2.)
