class NeuralNetwork{
	Matrix weights_ih;
	Matrix weights_ho;
	Matrix bias_h;
	Matrix bias_o;
	int input_nodes;
	int hidden_nodes;
	int output_nodes;
	float learning_rate;
	ActivationFunctions aFunction;



	NeuralNetwork(int input,int hidden,int output,float learning_rate){
		this.input_nodes=input;
		this.hidden_nodes=hidden;
		this.output_nodes=output;
		this.weights_ih=new Matrix(this.hidden_nodes,this.input_nodes).randomize(-1,1);
		this.weights_ho=new Matrix(this.output_nodes,this.hidden_nodes).randomize(-1,1);
		this.bias_h=new Matrix(this.hidden_nodes,1).randomize(-1,1);
		this.bias_o=new Matrix(this.output_nodes,1).randomize(-1,1);
		this.aFunction=new ActivationFunctions("sigmoid");
		this.learning_rate=learning_rate;
	}



	NeuralNetwork(JSONObject json){
		this.input_nodes=json.getInt("i");
		this.hidden_nodes=json.getInt("h");
		this.output_nodes=json.getInt("o");
		this.weights_ih=new Matrix(this.hidden_nodes,this.input_nodes).fromJSON(json.getJSONArray("w-ih"));
		this.weights_ho=new Matrix(this.output_nodes,this.hidden_nodes).fromJSON(json.getJSONArray("w-ho"));
		this.bias_h=new Matrix(this.hidden_nodes,1).fromJSON(json.getJSONArray("b-h"));
		this.bias_o=new Matrix(this.output_nodes,1).fromJSON(json.getJSONArray("b-o"));
		this.aFunction=new ActivationFunctions(json.getString("af"));
		this.learning_rate=json.getFloat("lr");
	}



	void draw(float[] in,int w,int h){
		Matrix input=new Matrix(1,this.input_nodes);
		input.from_array_data(in);
		input=input.transpose();
		Matrix hidden=input.mult(this.weights_ih);
		hidden.add(this.bias_h);
		hidden=this.aFunction.apply(hidden,"func");
		Matrix output=hidden.mult(this.weights_ho);
		output.add(this.bias_o);
		output=this.aFunction.apply(output,"func");
		ellipseMode(CENTER);
		textAlign(CENTER,CENTER);
		textFont(createFont("Consolas",13));
		for (int j=0; j<this.input_nodes; j++){
			for (int i=0; i<this.hidden_nodes; i++){
				stroke(map(abs(this.weights_ih.get(i,j)),0,max(this.weights_ih.get_max(),1),0,255));
				strokeWeight(this.weights_ih.get(i,j)<0?1.5:3);
				if (this.weights_ih.get(i,j)==0){
					stroke(230,35,35);
					strokeWeight(2.25);
				}
				line(1*(w/4.0),(j+1)*(h/(float)(this.input_nodes+1)),2*(w/4.0),(i+1)*(h/(float)(this.hidden_nodes+1)));
			}
		}
		for (int j=0; j<this.hidden_nodes; j++){
			for (int i=0; i<this.output_nodes; i++){
				stroke(map(abs(this.weights_ho.get(i,j)),0,max(this.weights_ho.get_max(),1),0,255));
				strokeWeight(this.weights_ho.get(i,j)<0?1.5:3);
				if (this.weights_ho.get(i,j)==0){
					stroke(230,35,35);
					strokeWeight(2.25);
				}
				line(2*(w/4.0),(j+1)*(h/(float)(this.hidden_nodes+1)),3*(w/4.0),(i+1)*(h/(float)(this.output_nodes+1)));
			}
		}
		strokeWeight(2);
		for (int i=0; i<this.input_nodes; i++){
			fill(map(input.get(i,0),0,1,0,255));
			stroke((map(input.get(i,0),0,1,0,255)+128)%255);
			circle(1*(w/4.0),(i+1)*(h/(float)(this.input_nodes+1)),40);
			fill(input.get(i,0)<0.5?255:0);
			text(nf(input.get(i,0),1,2),1*(w/4.0),(i+1)*(h/(float)(this.input_nodes+1)));
		}
		for (int i=0;i<this.hidden_nodes;i++){
			fill(map(hidden.get(i,0),0,1,0,255));
			stroke((map(hidden.get(i,0),0,1,0,255)+128)%255);
			circle(2*(w/4.0),(i+1)*(h/(float)(this.hidden_nodes+1)),40);
			fill(hidden.get(i,0)<0.5?255:0);
			text(nf(hidden.get(i,0),1,2),2*(w/4.0),(i+1)*(h/(float)(this.hidden_nodes+1)));
		}
		for (int i=0;i<this.output_nodes;i++){
			fill(map(output.get(i,0),0,1,0,255));
			stroke((map(output.get(i,0),0,1,0,255)+128)%255);
			circle(3*(w/4.0),(i+1)*(h/(float)(this.output_nodes+1)),40);
			fill(output.get(i,0)<0.5?255:0);
			text(nf(output.get(i,0),1,2),3*(w/4.0),(i+1)*(h/(float)(this.output_nodes+1)));
		}
	}



	float[] predict(float[] in){
		Matrix input=new Matrix(1,this.input_nodes);
		input.from_array_data(in);
		input=input.transpose();
		Matrix hidden=input.mult(this.weights_ih);
		hidden.add(this.bias_h);
		hidden=this.aFunction.apply(hidden,"func");
		Matrix output=hidden.mult(this.weights_ho);
		output.add(this.bias_o);
		output=this.aFunction.apply(output,"func");
		return output.toArray();
	}



	void train(float[] in,float[] o){
		Matrix input=new Matrix(1,this.input_nodes);
		input.from_array_data(in);
		input=input.transpose();
		Matrix hidden=input.mult(this.weights_ih);
		hidden.add(this.bias_h);
		hidden=this.aFunction.apply(hidden,"func");
		Matrix output=hidden.mult(this.weights_ho);
		output.add(this.bias_o);
		output=this.aFunction.apply(output,"func");
		Matrix tOutput=new Matrix(1,this.output_nodes);
		tOutput.from_array_data(o);
		tOutput=tOutput.transpose();
		Matrix oErrors=tOutput.sub(output).transpose();
		Matrix oGradient=this.aFunction.apply(output.transpose(),"deltaFunc");
		oGradient.multEl(oErrors);
		oGradient.multSc(this.learning_rate);
		Matrix weights_ho_delta=oGradient.mult(hidden).transpose();
		this.weights_ho.add(weights_ho_delta);
		this.bias_o.add(oGradient);
		Matrix hErrors=weights_ho.mult(oErrors);
		Matrix hGradient=this.aFunction.apply(hidden.transpose(),"deltaFunc");
		hGradient.multEl(hErrors);
		hGradient.multSc(this.learning_rate);
		Matrix weights_ih_delta=hGradient.mult(input);
		this.weights_ih.add(weights_ih_delta);
		this.bias_h.add(hGradient.transpose());
	}



	void train_multiple(float[][] in,float[][] out,int iterations){
		int last_proc=-1;
		for (int i=0; i<iterations; i++){
			if (int((float)i/iterations*100)!=last_proc){
				last_proc=int((float)i/iterations*100);
				//println(last_proc+"% complete...");
			}
			for (int j=0; j<in.length; j++){
				this.train(in[j],out[j]);
			}
		}
	}



	JSONObject toJSON(){
		JSONObject json=new JSONObject();
		json.setInt("i",this.input_nodes);
		json.setInt("h",this.hidden_nodes);
		json.setInt("o",this.output_nodes);
		json.setFloat("lr",this.learning_rate);
		json.setString("af",this.aFunction.t);
		json.setJSONArray("w-ih",this.weights_ih.toJSON());
		json.setJSONArray("w-ho",this.weights_ho.toJSON());
		json.setJSONArray("b-h",this.bias_h.toJSON());
		json.setJSONArray("b-o",this.bias_o.toJSON());
		return json;
	}
}
