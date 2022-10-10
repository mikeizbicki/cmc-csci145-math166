Due date: Tuesday, 18 October @ midnight

Instructions:

1. Download the jupyter notebook file.
   In order to run it, open a terminal in the same folder as the notebook and run the command
   ```
   $ OPENBLAS_NUM_THREADS=80 python3 -m jupyter notebook
   ```
   Instructions will appear on the screen for how to connect within firefox.

   > **NOTE:**
   > Some of the cells can take up to 15 minutes for an ordinary laptop to run.
   > Running on the lambda server, these cells finish in under a minute.
   > Therefore, I recommend running on the lambda server, but it is not required.
   > Running on the lambda server will require setting up port forwarding, which we will not cover in class.
   > The `OPENBLAS_NUM_THREADS=80` tells python to use up to 80 parallel processors when performing matrix calculations.

1. Complete each question directly inside the jupyter notebook,
   and upload the completed notebook to sakai.

1. There are 11 questions.
   The first 10 are worth 1 point each, and the last question (which requires programming) is worth 4 points,
   for 14 points total.
