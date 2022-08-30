```bash
#pip install -r requirements.txt
pip install -e .
# ./scripts/train.py --math_operator s5
math_operator=s5
. train.sh 50 $math_operator
# see train.sh for all the parameters
```
