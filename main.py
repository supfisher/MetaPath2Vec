from read_data import ReadData, Copus
from torch.utils.data import DataLoader
from SGNS import SGNS
from torch.optim import SparseAdam, lr_scheduler
from tqdm import tqdm
import torch as t


if __name__=="__main__":

    read_data = ReadData("/ibex/scratch/mag0a/Github/data/aminer.txt")
    dataset = Copus(read_data)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, collate_fn=dataset.collate)

    model = SGNS(100000000, 128, n_negs=20, weights=read_data.word_frequency)
    optimizer = SparseAdam(model.parameters(), lr=0.025)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))

    running_loss = 0.0

    for epoch in range(1):
        for i, (u, v) in enumerate(tqdm(dataloader)):
            scheduler.step()
            optimizer.zero_grad()
            loss = model(u, v)
            loss.backward()
            optimizer.step()

            running_loss = running_loss * 0.9 + loss.item() * 0.1
            if i > 0 and i % 500 == 0:
                print(" Loss: " + str(running_loss))

        t.save(model.state_dict(), 'model.pkl')
        model.save_embeddings()