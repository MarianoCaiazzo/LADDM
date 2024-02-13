import re

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