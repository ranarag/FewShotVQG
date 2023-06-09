import torch
import torch.nn as nn

class CategoryEncoder(nn.Module):
    def __init__(self, num_categories, embedding_dim, encoder_dim, embedding=None, scale_shift=True):
        super(CategoryEncoder, self).__init__()
        print(num_categories)
        print(embedding_dim)
        self.scale_shift = scale_shift
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        # self.category_encoder = MLP(embedding_dim, embedding_dim // 2, encoder_dim // 2, num_layers=3)
        # self.category_encoder = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim // 2),
        #     nn.BatchNorm1d(embedding_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(embedding_dim // 2, encoder_dim // 2),
        #     nn.BatchNorm1d(encoder_dim // 2),
        #     nn.ReLU(),

        #     # nn.Linear(embedding_dim // 2, encoder_dim // 2),
        #     # nn.BatchNorm1d(encoder_dim // 2),
        #     # nn.ReLU()
        # )

        # for nbn
        if embedding is not None:
            print("OKAY")
            self.embedding.weight = nn.Parameter(embedding, requires_grad=False)
        if self.scale_shift:
            self.category_encoder = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
#                 nn.BatchNorm1d(embedding_dim // 2),            
                nn.Linear(embedding_dim // 2, encoder_dim // 2),
                nn.ReLU()
#                 nn.BatchNorm1d(encoder_dim // 2)   
                # nn.Linear(embedding_dim // 2, encoder_dim // 2),
#                 nn.BatchNorm1d(encoder_dim // 2)
#                 nn.ReLU()
            )
            self.cat_wts = nn.Linear(encoder_dim //  2, encoder_dim)
            self.cat_biases = nn.Linear(encoder_dim //  2, encoder_dim)
        else:
            self.category_encoder = nn.Sequential(
                nn.Linear(embedding_dim, encoder_dim),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(embedding_dim // 2),            
#                 nn.Linear(embedding_dim // 2, encoder_dim),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(encoder_dim),   
                # nn.Linear(embedding_dim // 2, encoder_dim // 2),
#                 nn.BatchNorm1d(encoder_dim // 2),
                nn.ReLU()
            )           




    def forward(self, categories, weights=None):
        """Encode categories.
        Args:
            categories: Batch of answer Tensors.
        Returns:
            Batch of categories encoded into features.
        """
#         print(categories.size())
#         batch_size = categories.size(0)
        embedding_categories = self.embedding(categories)
#         embedding_categories = embedding_categories.view(batch_size, -1)
#         for i in range(categories.size(1)):
            
#             embedding_categories = torch.add(embedding_categories, self.embedding(categories[:, i]))
#         print(embedding_categories.size())
#         exit()
        if self.scale_shift:
            encoded_vals = self.category_encoder(embedding_categories)
            cat_wt_vals = self.cat_wts(encoded_vals)
            cat_b_vals = self.cat_biases(encoded_vals)
            return cat_wt_vals, cat_b_vals
        else:
            encoded_vals = self.category_encoder(embedding_categories)
            return encoded_vals
        # return encoded_vals
 
