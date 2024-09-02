import torch

# TODO: needs optimization, it does not support batched inputs and the iteration is not efficient. Also, it does not support better methods than greedy decoding.

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0, unk="?"):
        super().__init__()
        self.labels = labels
        self.blank = blank
        self.unk = unk

    def forward(self, emission: torch.Tensor, return_sequence=True, remove_consecutive_sil=False):
        """
        Args:
            emission (torch.Tensor): The emission probabilities, shape (T, C).
            return_sequence (bool): Whether to return the sequence or the map of the decoded sequence.
            The map is a boolean tensor with the same shape as the emission tensor, where the positions of the decoded sequence are set to True disregarding the blank token (silence).
            remove_consecutive_sil (bool): Whether to remove consecutive silence tokens in decoded map (only used if return_sequence is False).
        """
        assert len(emission.shape) == 2, "Does not support batched inputs, emission should be T x C."
        if remove_consecutive_sil and return_sequence:
            raise ValueError("collapse_silence should be used only if return_sequence is False.")
        indices = torch.argmax(emission, dim=-1) 
        if return_sequence:
            indices = torch.unique_consecutive(indices, dim=-1)
            indices = [i for i in indices if i != self.blank]
            output = []
            for i in indices:
                if int(i) < len(self.labels):
                    output.append(self.labels[i])
                else:
                    output.append(self.unk)
            return output
        else:
            map = torch.zeros_like(emission, dtype=torch.bool)
            b = None
            for i in range(len(indices)):
                if indices[i] != self.blank and b is None:
                    b = indices[i]
                if b != indices[i]:
                    map[i, indices[i]] = True
                else:
                    map[i, self.blank] = True
                b = indices[i]
            if remove_consecutive_sil: # remove consecutive silence
                map = torch.unique_consecutive(map, dim=0)
            return map
    
if __name__ == "__main__":
    labels = ["a", "b", "c"]
    decoder = GreedyCTCDecoder(labels)
    emission = torch.rand(10, 3)
    sequence = decoder(emission, return_sequence=True)
    print(sequence)
    map = decoder(emission, return_sequence=False)
    print(map.T.int())
    map = decoder(emission, return_sequence=False, remove_consecutive_sil=True)
    print(map.T.int())