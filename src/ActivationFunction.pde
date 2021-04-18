class ActivationFunctions{
	String t;



	ActivationFunctions(String t){
		this.t=t;
	}



	Matrix apply(Matrix m,String p){
		Matrix nm=new Matrix(m.w,m.h);
		for (int j=0; j<m.h; j++){
			for (int i=0; i<m.w; i++){
				if (this.t.equals("sigmoid")&&p.equals("func")){
					nm.set(i,j,this.sigmoid(m.get(i,j),0).x);
				}
				if (this.t.equals("sigmoid")&&p.equals("deltaFunc")){
					nm.set(i,j,this.sigmoid(0,m.get(i,j)).y);
				}
			}
		}
		return nm;
	}



	PVector sigmoid(float x,float y){
		return new PVector(1/(1+exp(-x)),y*(1-y));
	}
}
