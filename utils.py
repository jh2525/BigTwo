import torch
from typing import List, Dict

def concat_tensor(tensor_list: List[torch.Tensor]):
    sizes = torch.Tensor([tensor.shape for tensor in tensor_list]).long()
    if sizes.dim() == 1:
        return torch.concat(tensor_list)
    else:
        max_sizes = [v.item() for v in (sizes[:, 1:].max(dim=0)[0])]
        results = []
        for tensor in tensor_list:

            extended_tensor = torch.zeros((tensor.size(0), ) + tuple(max_sizes), dtype = tensor.dtype)
            target = extended_tensor
            for dim, size in enumerate(tensor.size()):
                target = target.narrow(dim, 0, size)

            target[:] = tensor
            results.append(extended_tensor)
    return torch.concat(results)



def concat_dict_tensors(tensor_dicts : List[Dict[str, torch.Tensor]]):
    result = {}

    keys = tensor_dicts[0].keys()
    for key in keys:
        result[key] = concat_tensor([dictionary[key] for dictionary in tensor_dicts])

    return result