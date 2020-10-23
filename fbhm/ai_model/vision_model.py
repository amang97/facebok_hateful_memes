class VisionModel:
    def __init__(self):
        super().__init__()
    
    def visual_embedding(self, img_paths, k):
        """
        Use pre-trained faster RCNN model on visual Genome to generate ROI in
        the images (k = # of desired regions)

        Assumes:
        ve is a np array
        """
        # TODO
        ve = None
        return ve