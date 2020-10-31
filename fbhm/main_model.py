from __future__ import print_function
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc_net import FC_Net  

class main_model(nn.Module): 
  def __init__(self, vision_feat, text_feat, attention, num_hidden): 
    
    self.vision_feat = vision_feat 
    self.text_feat = text_feat
    self.attention = attention 

    self.vision_fcnet = FC_Net([vision_feat.size(2), num_hidden], nn.ReLU())
    self.text_fcnet = FC_Net([text_feat.size(1), num_hidden], nn.ReLU) 

    self.classifier_layers = [
      weight_norm(nn.Linear(num_hidden, num_hidden),dim=None),
      nn.ReLU(),  
      weight_norm(nn.Linear(num_hidden, 2),dim=None),
    ] 

    self.classifier = nn.Sequential(*self.classifier_layers)

  def forward(vision_input, text_input): 
    '''
      text_input = [batches, text_feature_size] 
      vision_input = [batches, num_objects, vision_feature_size] 
    '''
    
    attention_weights = self.attention(vision_input, text_input) #[batches, num_obj] 
    vision_rep = (attention_weights * vision_input).sum(1)
    
    vision_rep = self.vision_fcnet(self.vision_rep) 
    text_rep = self.text_fcnet(self.text_feat) 

    combined_reb = vision_rep * text_rep 
    out = self.classifier(combined_rep) 

    return out 


def build_main_model(): 
  pass 
