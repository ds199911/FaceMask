import requests 

# https://your-heroku-app-name.herokuapp.com/predict
# http://localhost:5000/predict
resp = requests.post("https://testingapicv.herokuapp.com/predict", files={'file': open('mask1.jpeg', 'rb')})

print(resp.text)
