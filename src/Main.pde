import java.io.File;



final int LETTERS=10;
final int MODE_PREDICT=0;
final int MODE_TRAIN=1;
final int MODE_TEST=2;
final String[] ALPHABET={"a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"};
final String[] LANGS={"english","polish"};



String[][] LANGS_DICTS;
NeuralNetwork nn;



void setup(){
	int m=MODE_PREDICT;
	LANGS_DICTS=load_langs();
	File f=new File(dataPath("nn.json"));
	if (f.exists()&&f.isFile()){
		nn=new NeuralNetwork(loadJSONObject("nn.json"));
	}
	else{
		nn=new NeuralNetwork(LETTERS*ALPHABET.length,LETTERS,LANGS.length,0.001);
	}
	saveJSONObject(nn.toJSON(),"data/nn.json");
	if (m==MODE_PREDICT){
		String tw="Test";
		String l=to_lang(nn.predict(to_float(tw)));
		println("The word '"+tw+"' is "+l);
	}
	else if (m==MODE_TRAIN){
		test();
		int total=0;
		for (int i=0;i<LANGS_DICTS.length;i++){
			for (int j=0;j<LANGS_DICTS[i].length;j++){
				if (LANGS_DICTS[i][j]!=null){
					total++;
				}
			}
		}
		float[][] ins=new float[total][];
		float[][] outs=new float[total][];
		int idx=0;
		for (int i=0;i<LANGS_DICTS.length;i++){
			for (int j=0;j<LANGS_DICTS[i].length;j++){
				if (LANGS_DICTS[i][j]!=null){
					ins[idx]=to_float(LANGS_DICTS[i][j]);
					float[] o=new float[LANGS.length];
					o[i]=1;
					outs[idx]=o;
					idx++;
				}
			}
		}
		int t=millis();
		nn.train_multiple(ins,outs,1000);
		t=millis()-t;
		println(t);
		test();
	}
	else{
		test();
	}
	saveJSONObject(nn.toJSON(),"data/nn.json");
}



String[][] load_langs(){
	String[][] data=new String[LANGS.length][];
	int i=0;
	for (String l:LANGS){
		String[] a=loadStrings("lang/"+l+".txt");
		String[] b=new String[a.length-1];
		for (int j=0,k=0;j<a.length;j++){
			if (a[j].length()<=LETTERS){
				b[k++]=a[j].toLowerCase();
			}
			else{
				println("[WARN] The word '"+a[j].toLowerCase()+"' is too long ");
			}
		}
		data[i]=b;
		i++;
	}
	return data;
}



float[] to_float(String s){
	float[] data=new float[LETTERS*ALPHABET.length];
	for (int i=0;i<s.length();i++){
		for (int j=0;j<ALPHABET.length;j++){
			if (ALPHABET[j].equals(str(s.charAt(i)))){
				data[i*ALPHABET.length+j]=1;
				break;
			}
		}
	}
	return data;
}



String to_lang(float[] data){
	String s=LANGS[0];
	float b=data[0];
	for (int i=0;i<data.length;i++){
		if (data[i]>b){
			b=data[i];
			s=LANGS[i];
		}
	}
	return s;
}



void test(){
	int correct=0;
	int total=0;
	for (int i=0;i<LANGS_DICTS.length;i++){
		for (int j=0;j<LANGS_DICTS[i].length;j++){
			if (LANGS_DICTS[i][j]!=null){
				String l=to_lang(nn.predict(to_float(LANGS_DICTS[i][j])));
				if (l.equals(LANGS[i])){
					correct++;
				}
				total++;
			}
		}
	}
	println("The Neural Network is "+(float)correct/total*100+"% accurate");
}
