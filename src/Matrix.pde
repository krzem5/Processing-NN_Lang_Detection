class Matrix{
	int w,h;
	float[][] data;



	Matrix(int w,int h){
		this.w=w;
		this.h=h;
		this.data=new float[h][w];
		this.fill(0);
	}



	Matrix set(int i,int j,float v){
		this.data[j][i]=v;
		return this;
	}



	float get(int i,int j){
		return this.data[j][i];
	}



	float[][] get_all(){
		return this.data;
	}



	float[] toArray(){
		float[] a=new float[this.h];
		for (int j=0;j<this.h;j++){
			a[j]=this.data[j][0];
		}
		return a;
	}



	Matrix fill(float v){
		for (int j=0;j<this.h;j++){
			for (int i=0;i<this.w;i++){
				this.data[j][i]=v;
			}
		}
		return this;
	}



	Matrix from_array_data(float[] a){
		for (int j=0;j<this.h;j++){
			this.data[j][0]=a[j];
		}
		return this;
	}



	Matrix randomize(float v_min,float v_max){
		for (int j=0;j<this.h;j++){
			for (int i=0;i<this.w;i++){
				this.data[j][i]=random(1)*(v_max-v_min)+v_min;
			}
		}
		return this;
	}



	Matrix add(Matrix m){
		for (int j=0;j<this.h;j++){
			for (int i=0;i<this.w;i++){
				this.data[j][i]+=m.data[j][i];
			}
		}
		return this;
	}



	Matrix multEl(Matrix m){
		for (int j=0;j<this.h;j++){
			for (int i=0;i<this.w;i++){
				this.data[j][i]*=m.data[j][i];
			}
		}
		return this;
	}



	Matrix multSc(float sc){
		for (int j=0;j<this.h;j++){
			for (int i=0;i<this.w;i++){
				this.data[j][i]*=sc;
			}
		}
		return this;
	}



	Matrix sub(Matrix m){
		Matrix o=new Matrix(this.w,this.h);
		for (int j=0;j<this.h;j++){
			for (int i=0;i<this.w;i++){
				o.data[j][i]=this.data[j][i]-m.data[j][i];
			}
		}
		return o;
	}



	Matrix mult(Matrix m){
		if (this.w!=m.h){
			println("Columns A doesn't match rows B");
			this.log();
			m.log();
			return null;
		}
		Matrix nm=new Matrix(m.w,this.h);
		for (int i=0;i<this.h;i++){
			for (int j=0;j<m.w;j++){
				for (int k=0;k<this.w;k++){
					nm.data[i][j]+=this.data[i][k]*m.data[k][j];
				}
			}
		}
		return nm;
	}



	Matrix transpose(){
		Matrix m=new Matrix(this.h,this.w);
		for (int j=0;j<this.h;j++){
			for (int i=0;i<this.w;i++){
				m.data[i][j]=this.data[j][i];
			}
		}
		return m;
	}



	float get_max(){
		float max=-999999999;
		for (int j=0;j<this.h;j++){
			for (int i=0;i<this.w;i++){
				max=max(abs(this.data[j][i]),max);
			}
		}
		return max;
	}



	Matrix log(){
		String s="Matrix -> "+this.w+" x "+this.h+"\n";
		for (int i=0;i<("Matrix -> "+this.w+" x "+this.h+"\n").length();i++){
			s+="-";
		}
		int[] cw=new int[this.w];
		boolean[] mn=new boolean[this.w];
		for (int j=0;j<this.h;j++){
			for (int i=0;i<this.w;i++){
				cw[i]=max(cw[i],str(this.data[j][i]).replace("-","").length());
				mn[i]=mn[i]||false;
				mn[i]=str(str(this.data[j][i]).charAt(0))=="-"||mn[i];
			}
		}
		for (int j=0;j<this.h;j++){
			String str="";
			for (int i=0;i<this.w;i++){
				str+=" ";
				if (mn[i]==true&&str(str(this.data[j][i]).charAt(0))!="-"){
					str+=" ";
				}
				str+=this.data[j][i];
				for (int k=cw[i]-str(this.data[j][i]).replace("-","").length();k>=0;k--){
					str+=" ";
				}
			}
			s+="\n"+str.substring(1);
		}
		print(s+"\n\n");
		return this;
	}



	JSONArray toJSON(){
		JSONArray json=new JSONArray();
		for (int j=0;j<this.h; j++){
			JSONArray row=new JSONArray();
			for (int i=0; i<this.w; i++){
				row.setFloat(i,this.data[j][i]);
			}
			json.setJSONArray(j,row);
		}
		return json;
	}



	Matrix fromJSON(JSONArray json){
		for (int j=0; j<this.h; j++){
			JSONArray row=json.getJSONArray(j);
			for (int i=0; i<this.w; i++){
				this.data[j][i]=row.getFloat(i);
			}
		}
		return this;
	}
}
