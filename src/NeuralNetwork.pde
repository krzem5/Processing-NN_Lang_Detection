class Matrix {
  int w, h;
  float[][] data;
  Matrix(int w, int h) {
    this.w=w;
    this.h=h;
    this.data=new float[h][w];
    this.fill(0);
  }
  Matrix set(int i, int j, float v) {
    this.data[j][i]=v;
    return this;
  }
  float get(int i, int j) {
    return this.data[j][i];
  }
  float[][] get_all() {
    return this.data;
  }
  float[] toArray() {
    float[] a=new float[this.h];
    for (int j=0; j<this.h; j++) {
      a[j]=this.data[j][0];
    }
    return a;
  }
  Matrix fill(float v) {
    for (int j=0; j<this.h; j++) {
      for (int i=0; i<this.w; i++) {
        this.data[j][i]=v;
      }
    }
    return this;
  }
  Matrix from_array_data(float[] a) {
    for (int j=0; j<this.h; j++) {
      this.data[j][0]=a[j];
    }
    return this;
  }
  Matrix randomize(float v_min, float v_max) {
    for (int j=0; j<this.h; j++) {
      for (int i=0; i<this.w; i++) {
        this.data[j][i]=random(1)*(v_max-v_min)+v_min;
      }
    }
    return this;
  }
  Matrix add(Matrix m) {
    for (int j=0; j<this.h; j++) {
      for (int i=0; i<this.w; i++) {
        this.data[j][i]+=m.data[j][i];
      }
    }
    return this;
  }
  Matrix multEl(Matrix m) {
    for (int j=0; j<this.h; j++) {
      for (int i=0; i<this.w; i++) {
        this.data[j][i]*=m.data[j][i];
      }
    }
    return this;
  }
  Matrix multSc(float sc) {
    for (int j=0; j<this.h; j++) {
      for (int i=0; i<this.w; i++) {
        this.data[j][i]*=sc;
      }
    }
    return this;
  }
  Matrix sub(Matrix m) {
    Matrix o=new Matrix(this.w, this.h);
    for (int j=0; j<this.h; j++) {
      for (int i=0; i<this.w; i++) {
        o.data[j][i]=this.data[j][i]-m.data[j][i];
      }
    }
    return o;
  }
  Matrix mult(Matrix m) {
    if (this.w!=m.h) {
      println("Columns A doesn't match rows B");
      this.log();
      m.log();
      return null;
    }
    Matrix nm=new Matrix(m.w, this.h);
    for (int i=0; i<this.h; i++) {
      for (int j=0; j<m.w; j++) {
        for (int k=0; k<this.w; k++) {
          nm.data[i][j]+=this.data[i][k]*m.data[k][j];
        }
      }
    }
    return nm;
  }
  Matrix transpose() {
    Matrix m=new Matrix(this.h, this.w);
    for (int j=0; j<this.h; j++) {
      for (int i=0; i<this.w; i++) {
        m.data[i][j]=this.data[j][i];
      }
    }
    return m;
  }
  float get_max() {
    float max=-999999999;
    for (int j=0; j<this.h; j++) {
      for (int i=0; i<this.w; i++) {
        max=max(abs(this.data[j][i]), max);
      }
    }
    return max;
  }
  Matrix log() {
    String s="Matrix -> "+this.w+" x "+this.h+"\n";
    for (int i=0; i<("Matrix -> "+this.w+" x "+this.h+"\n").length(); i++) {
      s+="-";
    }
    int[] cw=new int[this.w];
    boolean[] mn=new boolean[this.w];
    for (int j=0; j<this.h; j++) {
      for (int i=0; i<this.w; i++) {
        cw[i]=max(cw[i], str(this.data[j][i]).replace("-", "").length());
        mn[i]=mn[i]||false;
        mn[i]=str(str(this.data[j][i]).charAt(0))=="-"||mn[i];
      }
    }
    for (int j=0; j<this.h; j++) {
      String str="";
      for (int i=0; i<this.w; i++) {
        str+=" ";
        if (mn[i]==true&&str(str(this.data[j][i]).charAt(0))!="-") {
          str+=" ";
        }
        str+=this.data[j][i];
        for (int k=cw[i]-str(this.data[j][i]).replace("-", "").length(); k>=0; k--) {
          str+=" ";
        }
      }
      s+="\n"+str.substring(1);
    }
    print(s+"\n\n");
    return this;
  }
  JSONArray toJSON() {
    JSONArray json=new JSONArray();
    for (int j=0; j<this.h; j++) {
      JSONArray row=new JSONArray();
      for (int i=0; i<this.w; i++) {
        row.setFloat(i, this.data[j][i]);
      }
      json.setJSONArray(j, row);
    }
    return json;
  }
  Matrix fromJSON(JSONArray json) {
    for (int j=0; j<this.h; j++) {
      JSONArray row=json.getJSONArray(j);
      for (int i=0; i<this.w; i++) {
        this.data[j][i]=row.getFloat(i);
      }
    }
    return this;
  }
}
class ActivationFunctions {
  String t;
  ActivationFunctions(String t) {
    this.t=t;
  }
  Matrix apply(Matrix m, String p) {
    Matrix nm=new Matrix(m.w, m.h);
    for (int j=0; j<m.h; j++) {
      for (int i=0; i<m.w; i++) {
        if (this.t.equals("sigmoid")&&p.equals("func")) {
          nm.set(i, j, this.sigmoid(m.get(i, j), 0).x);
        }
        if (this.t.equals("sigmoid")&&p.equals("deltaFunc")) {
          nm.set(i, j, this.sigmoid(0, m.get(i, j)).y);
        }
      }
    }
    return nm;
  }
  PVector sigmoid(float x, float y) {
    return new PVector(1/(1+exp(-x)), y*(1-y));
  }
}
class NeuralNetwork {
  Matrix weights_ih, weights_ho;
  Matrix bias_h, bias_o;
  int input_nodes, hidden_nodes, output_nodes;
  float learning_rate;
  ActivationFunctions aFunction;
  NeuralNetwork(int input, int hidden, int output, float learning_rate) {
    this.input_nodes=input;
    this.hidden_nodes=hidden;
    this.output_nodes=output;
    this.weights_ih=new Matrix(this.hidden_nodes, this.input_nodes).randomize(-1, 1);
    this.weights_ho=new Matrix(this.output_nodes, this.hidden_nodes).randomize(-1, 1);
    this.bias_h=new Matrix(this.hidden_nodes, 1).randomize(-1, 1);
    this.bias_o=new Matrix(this.output_nodes, 1).randomize(-1, 1);
    this.aFunction=new ActivationFunctions("sigmoid");
    this.learning_rate=learning_rate;
  }
  NeuralNetwork(JSONObject json) {
    this.input_nodes=json.getInt("i");
    this.hidden_nodes=json.getInt("h");
    this.output_nodes=json.getInt("o");
    this.weights_ih=new Matrix(this.hidden_nodes, this.input_nodes).fromJSON(json.getJSONArray("w-ih"));
    this.weights_ho=new Matrix(this.output_nodes, this.hidden_nodes).fromJSON(json.getJSONArray("w-ho"));
    this.bias_h=new Matrix(this.hidden_nodes, 1).fromJSON(json.getJSONArray("b-h"));
    this.bias_o=new Matrix(this.output_nodes, 1).fromJSON(json.getJSONArray("b-o"));
    this.aFunction=new ActivationFunctions(json.getString("af"));
    this.learning_rate=json.getFloat("lr");
  }
  float[] predict(float[] in) {
    Matrix input=new Matrix(1, this.input_nodes);
    input.from_array_data(in);
    input=input.transpose();
    Matrix hidden=input.mult(this.weights_ih);
    hidden.add(this.bias_h);
    hidden=this.aFunction.apply(hidden, "func");
    Matrix output=hidden.mult(this.weights_ho);
    output.add(this.bias_o);
    output=this.aFunction.apply(output, "func");
    return output.toArray();
  }
  void train(float[] in, float[] o) {
    Matrix input=new Matrix(1, this.input_nodes);
    input.from_array_data(in);
    input=input.transpose();
    Matrix hidden=input.mult(this.weights_ih);
    hidden.add(this.bias_h);
    hidden=this.aFunction.apply(hidden, "func");
    Matrix output=hidden.mult(this.weights_ho);
    output.add(this.bias_o);
    output=this.aFunction.apply(output, "func");

    Matrix tOutput=new Matrix(1, this.output_nodes);
    tOutput.from_array_data(o);
    tOutput=tOutput.transpose();

    Matrix oErrors=tOutput.sub(output).transpose();
    Matrix oGradient=this.aFunction.apply(output.transpose(), "deltaFunc");
    oGradient.multEl(oErrors);
    oGradient.multSc(this.learning_rate);
    Matrix weights_ho_delta=oGradient.mult(hidden).transpose();
    this.weights_ho.add(weights_ho_delta);
    this.bias_o.add(oGradient.transpose());

    Matrix hErrors=weights_ho.mult(oErrors);
    Matrix hGradient=this.aFunction.apply(hidden.transpose(), "deltaFunc");
    hGradient.multEl(hErrors);
    hGradient.multSc(this.learning_rate);
    Matrix weights_ih_delta=hGradient.mult(input);
    this.weights_ih.add(weights_ih_delta.transpose());
    this.bias_h.add(hGradient.transpose());
  }
  void train_multiple(float[][] in, float[][] out, int iterations, boolean log) {
    int last_proc=-1;
    for (int i=0; i<iterations; i++) {
      if (log==true) {
        if (int((float)i/iterations*100)!=last_proc) {
          last_proc=int((float)i/iterations*100);
          println(last_proc+"% complete...");
        }
      }
      for (int j=0; j<in.length; j++) {
        this.train(in[j], out[j]);
      }
    }
  }
  JSONObject toJSON() {
    JSONObject json=new JSONObject();
    json.setInt("i", this.input_nodes);
    json.setInt("h", this.hidden_nodes);
    json.setInt("o", this.output_nodes);
    json.setFloat("lr", this.learning_rate);
    json.setString("af", this.aFunction.t);
    json.setJSONArray("w-ih", this.weights_ih.toJSON());
    json.setJSONArray("w-ho", this.weights_ho.toJSON());
    json.setJSONArray("b-h", this.bias_h.toJSON());
    json.setJSONArray("b-o", this.bias_o.toJSON());
    return json;
  }
}
void graph_nn(NeuralNetwork nn, float[] in, int W, int H) {
  Matrix input=new Matrix(1, nn.input_nodes);
  input.from_array_data(in);
  input=input.transpose();
  Matrix hidden=input.mult(nn.weights_ih);
  hidden.add(nn.bias_h);
  hidden=nn.aFunction.apply(hidden, "func");
  Matrix output=hidden.mult(nn.weights_ho);
  output.add(nn.bias_o);
  output=nn.aFunction.apply(output, "func");
  ellipseMode(CENTER);
  textAlign(CENTER, CENTER);
  textFont(createFont("Consolas", 13));
  for (int j=0; j<nn.input_nodes; j++) {
    for (int i=0; i<nn.hidden_nodes; i++) {
      stroke(map(abs(nn.weights_ih.get(i, j)), 0, max(nn.weights_ih.get_max(), 1), 0, 255));
      strokeWeight(nn.weights_ih.get(i, j)<0?1.5:3);
      if (nn.weights_ih.get(i, j)==0) {
        stroke(230, 35, 35);
        strokeWeight(2.25);
      }
      line(1*(W/4.0), (j+1)*(H/(float)(nn.input_nodes+1)), 2*(W/4.0), (i+1)*(H/(float)(nn.hidden_nodes+1)));
    }
  }
  for (int j=0; j<nn.hidden_nodes; j++) {
    for (int i=0; i<nn.output_nodes; i++) {
      stroke(map(abs(nn.weights_ho.get(i, j)), 0, max(nn.weights_ho.get_max(), 1), 0, 255));
      strokeWeight(nn.weights_ho.get(i, j)<0?1.5:3);
      if (nn.weights_ho.get(i, j)==0) {
        stroke(230, 35, 35);
        strokeWeight(2.25);
      }
      line(2*(W/4.0), (j+1)*(H/(float)(nn.hidden_nodes+1)), 3*(W/4.0), (i+1)*(H/(float)(nn.output_nodes+1)));
    }
  }
  strokeWeight(2);
  for (int i=0; i<nn.input_nodes; i++) {
    fill(map(input.get(i, 0), 0, 1, 0, 255));
    stroke((map(input.get(i, 0), 0, 1, 0, 255)+128)%255);
    circle(1*(W/4.0), (i+1)*(H/(float)(nn.input_nodes+1)), 40);
    fill(input.get(i, 0)<0.5?255:0);
    text(nf(input.get(i, 0), 1, 2), 1*(W/4.0), (i+1)*(H/(float)(nn.input_nodes+1)));
  }
  for (int i=0; i<nn.hidden_nodes; i++) {
    fill(map(hidden.get(i, 0), 0, 1, 0, 255));
    stroke((map(hidden.get(i, 0), 0, 1, 0, 255)+128)%255);
    circle(2*(W/4.0), (i+1)*(H/(float)(nn.hidden_nodes+1)), 40);
    fill(hidden.get(i, 0)<0.5?255:0);
    text(nf(hidden.get(i, 0), 1, 2), 2*(W/4.0), (i+1)*(H/(float)(nn.hidden_nodes+1)));
  }
  for (int i=0; i<nn.output_nodes; i++) {
    fill(map(output.get(i, 0), 0, 1, 0, 255));
    stroke((map(output.get(i, 0), 0, 1, 0, 255)+128)%255);
    circle(3*(W/4.0), (i+1)*(H/(float)(nn.output_nodes+1)), 40);
    fill(output.get(i, 0)<0.5?255:0);
    text(nf(output.get(i, 0), 1, 2), 3*(W/4.0), (i+1)*(H/(float)(nn.output_nodes+1)));
  }
}
