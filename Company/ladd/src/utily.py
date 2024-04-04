import re
import torch

def regex_pattern(testo, pattern):
  # Trovare corrispondenze nel testo usando il pattern regex
  corrispondenze = re.search(pattern, testo)
  # print(corrispondenze)
  # Se ci sono corrispondenze, estrarre la prima parola e il numero dopo di essa
  if corrispondenze:
      action = corrispondenze.group(1)
      n_bytes = corrispondenze.group(5)
      # print(action, n_bytes)
      return action, n_bytes
  return None, None

PATTERN = r'(\w+)(\s)(\w+)(\s)(\w+)'

def normalize_tensor(tensor_data):
  tensor_min = torch.min(tensor_data)
  tensor_max = torch.max(tensor_data)

  # Normalizza il Tensor nell'intervallo [0, 1]
  normalized_tensor = (tensor_data - tensor_min) / (tensor_max - tensor_min)

  # Puoi anche normalizzare in un intervallo specifico, ad esempio [0, 99]
  new_min = 0
  new_max = 99
  normalized_tensor_specific_range = \
   (tensor_data - tensor_min) / (tensor_max - tensor_min) \
        * (new_max - new_min) + new_min

  return normalized_tensor_specific_range.to(torch.long)
