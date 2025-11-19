from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    # Your code here
    best = val_losses[0]
    count = 0
    i=0
    while(count<patience and i < len(val_losses)-1):
        if val_losses[i + 1] < best - min_delta:
            best = val_losses[i + 1]
            count = 0
        
        elif count < patience:
            count += 1 
        
        if count == patience:
            return (i + 1, val_losses.index(best))
        i+=1
    
    return (len(val_losses) - 1, val_losses.index(best))



