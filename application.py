from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Soil_Type=float(request.form.get('Soil_Type')),
            Sunlight_Hours = float(request.form.get('Sunlight_Hours')),
            Water_Frequency = float(request.form.get('Water_Frequency')),
            Fertilizer_Type = float(request.form.get('Fertilizer_Type')),
            Temperature = float(request.form.get('Temperature')),
            Humidity = float(request.form.get('Humidity'))
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=pred 

        return render_template('form.html',final_result=results)
    

if __name__ =='__main__':
        app.run(host='0.0.0.0',port=5000,debug=True)