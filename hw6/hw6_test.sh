wget -O rnn_1 https://www.csie.ntu.edu.tw/~b06902066/rnn_1
wget -O rnn_2 https://www.csie.ntu.edu.tw/~b06902066/rnn_2
wget -O rnn_3 https://www.csie.ntu.edu.tw/~b06902066/rnn_3
wget -O rnn_4 https://www.csie.ntu.edu.tw/~b06902066/rnn_4
wget -O rnn_5 https://www.csie.ntu.edu.tw/~b06902066/rnn_5
wget -O word2vec_test https://www.csie.ntu.edu.tw/~b06902066/word2vec_test
python submit_ensenble.py "$1" "$2" "word2vec_test" "$3" "rnn_1" "rnn_2" "rnn_3" "rnn_4" "rnn_5"
