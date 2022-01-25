def iou(output,target,eps=1e-5,threshold=0.5,use_sigmoid='True'):
    target_0=(target==0).float()
    target_255=(target==255).float()
    if use_sigmoid:
      output_0=1-output
      output_1=output
    else:
      output_0=output[:,0,...]
      output_1=output[:,1,...]
    
    output_0=(output_0>threshold).float()
    output_1=(output_1>threshold).float()
    
    intersection_0=((output_0*target_0).sum())+eps
    union_0=(output_0.sum()+target_0.sum()-intersection_0)+eps

    intersection_255=((output_1*target_255).sum())+eps
    union_255=(output_1.sum()+target_255.sum()-intersection_255)+eps
    
    score_0=intersection_0/union_0
    score_255=intersection_255/union_255
    
    return (score_0+score_255)/2.