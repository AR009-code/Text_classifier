
from flask import Flask, render_template,redirect, request
import testing_model as test_m
import training_model as train_m
import os, threading

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    result={"s_code":0,"status":"none"}
    return render_template('index.html',s_code=result['s_code'], status=result['status'],textArea="")

@app.route('/', methods=['POST'])
def get_text():
    text = request.form['text']

    if(request.form.get('action1')=='submit' and len(text)>0):

        (result, mtpc, mtnc, word_count, tm, t_log_p, t_log_n)=test_m.run_model([text])

        output={"s_code":1,"status":result, "trained_words":train_m.word_to_index,"log_p":train_m.log_p_positive,"log_n":train_m.log_p_negative,
                "mtpc":mtpc, "mtnc":mtnc, "wc":word_count, "t_matrix":tm, "t_log_p":t_log_p, "t_log_n":t_log_n}
 
        return render_template('index.html',s_code=output['s_code'], status=output['status'],tw=output['trained_words'], log_p=output['log_p'], 
                              log_n=output['log_n'], mtpc=output['mtpc'], mtnc=output['mtnc'], wc=output['wc'], t_matrix=output['t_matrix'],
                             t_log_p=output['t_log_p'], t_log_n=output['t_log_n'], textArea=text)

    elif(request.form.get('action2')=='reset'):
        return redirect('/')
    else:
        output={"s_code":-1,"status":"Write something in the textbox, Before submit"}
        return render_template('index.html',s_code=output['s_code'], status=output['status'],textArea=text)

@app.route('/old_project',methods=['GET'])
def old_project():
    t1=threading.Thread(None,run_old_project,"django process")
    t1.start()
    return "executed"

def run_old_project():
    os.system('cmd /c "cd .. & cd .\Sentimental_analysis-Sem-7\ & .\start.bat"')
    



if __name__ == '__main__':
    app.run(debug=True)
