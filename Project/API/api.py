from flask import Flask
from flask_restx import Api, Resource, fields
from sklearn.externals import joblib
from proyecto3 import predict_proba

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Movie genre classification',
    description='<center><img src="https://serea2017.uniandes.edu.co/images/Logo.png" height="60" width="200" align="Center"/><br>by:<br>Nelson Aldana Mart&iacute;nez - 201924128<br>Sergio Alberto Mora Pardo - 201920547<br>Jaime Orjuela Viracach&aacute; - 201924252<br>Juan Sebasti&aacute;n Rinc&oacute;n - 201214767</center>')
ns = api.namespace('predict', 
    description='Movie genre classification')
   
parser = api.parser()

parser.add_argument(
    'plot', 
    type=str, 
    required=True, 
    help='Movie plot', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String
})

@ns.route('/')
class Predict(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        plot = args['plot']
        
        return {
         "result": predict_proba(plot)
        }, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)


