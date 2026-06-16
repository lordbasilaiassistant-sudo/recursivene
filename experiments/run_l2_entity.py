"""Wake Nous-L2 and let it live in a multi-dimensional world. Run twice -> it resumes.
Run:  python experiments/run_l2_entity.py
"""

from _util import REPO_ROOT
from recursivene.entity_l2 import L2Entity


def main():
    L2Entity(name="NousL2", home=REPO_ROOT, dim=3, seed=7).live(seasons=18)


if __name__ == "__main__":
    main()
