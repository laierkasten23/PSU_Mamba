from segment_anything import sam_model_registry, SamPredictor


def getSAMPredictor(sam_model="base"):
    if sam_model == "base":
        return SamPredictor(
            sam_model_registry["vit_b"](
                checkpoint="../KI-Morph_Container/fakeSDS/model/SAM/sam_vit_b_01ec64.pth"
            )
        )
    elif sam_model == "large":
        return SamPredictor(
            sam_model_registry["vit_l"](
                checkpoint="../KI-Morph_Container/fakeSDS/model/SAM/sam_vit_l_0b3195.pth"
            )
        )
    elif sam_model == "huge":
        return SamPredictor(
            sam_model_registry["vit_h"](
                checkpoint="../KI-Morph_Container/fakeSDS/model/SAM/sam_vit_h_4b8939.pth"
            )
        )
    else:
        raise ValueError("sam_model must be one of 'base', 'large', 'huge'")
