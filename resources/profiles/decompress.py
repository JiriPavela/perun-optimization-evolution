import os
import json
import pickle
import bz2
from typing import Dict, List, Any, Optional


# JSON type alias
JsonType = Dict[str, Any]


with open(os.path.join('profiles', 'ccsds_100.pbz2'), 'rb') as compressed:
    print(compressed.name)
    profile = pickle.loads(bz2.decompress(compressed.read()))

# tid -> caller -> uid -> {'e': [amounts], 'i': [amounts], 'o': [call order]}
tids = list(profile.keys())
profile = profile[tids[0]]

callers = set(profile.keys())
uids = set()
for caller, callees in profile.items():
    # print(f'{caller}: {uids.keys()}')
    uids |= set(callees.keys())

print(f'UIDS / CALLERS = {uids - callers}')


# callers = list(profile[tids[0]].keys())



# print(profile[tids[0]]['!root!'])
# print(tids)
# print(callers)

    
