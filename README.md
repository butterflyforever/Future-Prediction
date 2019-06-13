# Events-Embedding-and-Future-price-prediction

### Environment

```
python 3.5
keras
json
urllib
theano
LTP(https://ltp.readthedocs.io/zh_CN/latest/ltpserver.html)

```

### Data preparation for getting EventEmbedding
```
python prepare/sina.py              // Get sina_news from /home/haijun/project/causality/crawl/sina-crawler/data
python prepare/extract_event.py     // Parse the sentences and use @ to divide them (Time consuming!!!).
                                    // Open LTP ports first!!! The result is saved as "all_events_15to18.txt".
                                    // You have to Download LTP NLP tool https://ltp.readthedocs.io/zh_CN/latest/ltpserver.html 
python prepare/filter_events.py     // Remove stopwords and some noise events. The result is saved as "less_noise_all_events_15to18.txt"
```
### EventEmbedding training and generating

```
python prepare/extractVocab.py      // Convert all Events to index-word for training
python embedding/eventEmbedding.py  // Get the event embedding. The model is saved as "TrainedParams.pickle"
                                    // The event embedding is saved as "resultEmbeding_all.pickle"
```
### Data preparation for prediction
```
python prepare/prepare_for_prediction.py    // Save the wordEmbedding and Eventembedding of Training data 
                                            // in "/data/all/1000/model/dayemb/" (only 17 types)
python prepare/BoW_embedding.py             // Save the BoW_embedding of Training/Test data 
                                            // in "/data/all/1000/model/dayemb/" (only 17 types)
python prepare/E_NN_embedding.py            // Save the E_NN_embedding of Training/Test data 
                                            // in "/data/all/1000/model/dayemb/" (only 17 types)
python prepare/prepare_for_prediction.py    // Save the wordEmbedding and Eventembedding of Training data 
                                            // in "/data/all/1000/model/dayemb/" (only 17 types)

```
### Prediction
```
python prediction/EB_predict_nn.py      // EBNN model
python prediction/EB_prediction_cnn.py  // EBCNN model
python prediction/WB_predict_nn.py      // WBNN model
python prediction/WB_prediction_cnn.py  // WBCNN model

```

The data saving directories are really unordered..... 

If I have time, I will clean it anyway.


