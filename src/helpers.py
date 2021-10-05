from typing import List


class Helper():
    """
    Performs a series of transforms and calculations, depending on necessity

    parameters :
    """

    @staticmethod
    def flatten_list(list) -> List:
        return [item for sublist in list for item in sublist]