# Script that runs run.py 3 times (for each model)

import sys
from run import main

if __name__ == "__main__":
    sys.argv = ["run.py", "--models", "qwen"]
    main()

    sys.argv = ["run.py", "--models", "gemma3"]
    main()

    sys.argv = ["run.py", "--models", "llama"]
    main()
