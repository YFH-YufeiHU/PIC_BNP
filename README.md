# PIC_BNP
In order to launch this code, you should run the following commands.
~~~bash
git clone -b remove_torch_save https://github.com/NielsRogge/unilm.git
pip install unilm/layoutlm
git clone https://github.com/huggingface/transformers
pip install ./transformers
~~~

In addition, you should install other packages which are:
* matplotlib
* torch
* pip

After these, you can do the training with the following command.
~~~bash
python main.py
~~~

Similarly, you can do the evaluation after training.
~~~bash
python evaluate.py
~~~
