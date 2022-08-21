import requests

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"



#Call each API endpoint and store the responses
response1 = requests.get('http://127.0.0.1:8000/prediction?dfpath=ingesteddata&dfname=finaldata.csv').status_code
response2 = requests.get('http://127.0.0.1:8000/scoring?test_data_path=testdata&model_path=practicemodels').status_code
response3 = requests.get('http://127.0.0.1:8000/summarystats').status_code
response4 = requests.get('http://127.0.0.1:8000/diagnostics').status_code

#combine all API responses
responses = [str(response1), str(response2), str(response3), str(response4)]

#write the responses to your workspace
with open('apicallstatus.txt', 'w') as file:
    file.write(','.join(responses))