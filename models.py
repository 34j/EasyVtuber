import torch.nn as nn
import torch
from tha2.poser.modes.mode_20 import load_face_morpher, load_face_rotater, load_combiner
from hashlib import sha256
from cm_time import timer
import numpy as np
from tqdm import tqdm

mouth_eye_variable_index = [2, 3, 14, 25, 26]
pose_vector_search_space = (np.linspace(-0.1, 0.1, 3), np.linspace(-0.6, 0.6, 15), np.array([0]))
mouth_eye_vector_search_space = np.linspace(0., 1., 3)


class TalkingAnimeLightCached(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, image: torch.Tensor, mouth_eye_vector: torch.Tensor, pose_vector: torch.Tensor) -> torch.Tensor:
        x, y, z = pose_vector[0].tolist()
        # find nearest
        x = pose_vector_search_space[0][np.abs(pose_vector_search_space[0] - x).argmin()]
        y = pose_vector_search_space[1][np.abs(pose_vector_search_space[1] - y).argmin()]
        z = pose_vector_search_space[2][np.abs(pose_vector_search_space[2] - z).argmin()]
        base = f"search/2-{x:g}-{y:g}-{z:g}-0"
        base_out = torch.load(base + ".pt")
        final_out = base_out.clone()
        for i in mouth_eye_variable_index:
            s = mouth_eye_vector[0, i].item()
            s = mouth_eye_vector_search_space[np.abs(mouth_eye_vector_search_space - s).argmin()]
            out = torch.load(f"search/{i:g}-{x:g}-{y:g}-{z:g}-{s:g}.pt")
            final_out += out - base_out
        return final_out

class TalkingAnimeLight(nn.Module):
    def __init__(self):
        super(TalkingAnimeLight, self).__init__()
        self.face_morpher = load_face_morpher('pretrained/face_morpher.pt')
        self.two_algo_face_rotator = load_face_rotater('pretrained/two_algo_face_rotator.pt')
        self.combiner = load_combiner('pretrained/combiner.pt')

    def forward(self, image, mouth_eye_vector, pose_vector):
        x = image.clone()
        
        with timer() as t:
            mouth_eye_morp_image = self.face_morpher(image[:, :, 32:224, 32:224], mouth_eye_vector)
        # print(f"morpher took {t.elapsed}s")
        
        x[:, :, 32:224, 32:224] = mouth_eye_morp_image
        
        with timer() as t:
            rotate_image = self.two_algo_face_rotator(x, pose_vector)[:2]
        # print(f"rotator took {t.elapsed}s")
        
        with timer() as t:
            output_image = self.combiner(rotate_image[0], rotate_image[1], pose_vector)
        # print(f"combiner took {t.elapsed}s")
        return output_image


class TalkingAnime(nn.Module):
    def __init__(self):
        super(TalkingAnime, self).__init__()

    def forward(self, image, mouth_eye_vector, pose_vector):
        x = image.clone()
        mouth_eye_morp_image = self.face_morpher(image[:, :, 32:224, 32:224], mouth_eye_vector)
        x[:, :, 32:224, 32:224] = mouth_eye_morp_image
        rotate_image = self.two_algo_face_rotator(x, pose_vector)[:2]
        output_image = self.combiner(rotate_image[0], rotate_image[1], pose_vector)
        return output_image
    
    
# repeat n times to check batch inference performance
# n = 20
# image = image.repeat(n, 1, 1, 1)
# mouth_eye_vector = mouth_eye_vector.repeat(n, 1)
# pose_vector = pose_vector.repeat(n, 1)

def infer_all(image: torch.Tensor) -> None:
    model = TalkingAnimeLight().to("cuda").eval()
    
    spaces = []
    for i in mouth_eye_variable_index:
        for x in pose_vector_search_space[0]:
            for y in pose_vector_search_space[1]:
                for z in pose_vector_search_space[2]:
                    for s in mouth_eye_vector_search_space:
                        spaces.append((i, x, y, z, s))
                       
    pbar = tqdm(spaces)
    batch_size = 20
    space_chunks = np.array_split(spaces, len(spaces) // batch_size + 1)
    for space_chunk in space_chunks:
        space_chunk = np.array(space_chunk)
        pose_vector = torch.tensor(space_chunk[:, 1:4]).to("cuda", dtype=torch.float32)
        mouth_eye_vector = torch.zeros(len(space_chunk), 27).to("cuda", dtype=torch.float32)
        for i, _, _, _, s in space_chunk:
            mouth_eye_vector[0, int(i)] = s
        with torch.no_grad():
            out = model(image.repeat(len(space_chunk), 1, 1, 1)
                        , mouth_eye_vector, pose_vector).detach().cpu()
        for i, x, y, z, s in space_chunk:
            i = int(i)
            pbar.set_description(f"i={i:g}, x={x:g}, y={y:g}, z={z:g}, s={s:g}")
            torch.save(out[i], f"search/{i:g}-{x:g}-{y:g}-{z:g}-{s:g}.pt")
        pbar.update(len(space_chunk))
        
        
if __name__ == "__main__":
    from PIL import Image
    from utils import preprocessing_image
    img = Image.open(f"character/0001.png").resize((256, 256))
    input_image = preprocessing_image(img).unsqueeze(0).to("cuda")
    infer_all(input_image)