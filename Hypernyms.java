

import java.util.HashMap;
import java.util.HashSet;

import rita.RiWordNet;

public class Hypernyms {
	public static RiWordNet wordnet = new RiWordNet(
			"thirdparty/WordNet/2.1/dict");

	public static PorterStemmer stemmer = new PorterStemmer();

	public static String getStem(String token) {
		String stemsPart1 = stemmer.step1(token);
		String stemsPart2 = stemmer.step2(stemsPart1);
		String stemsPart3 = stemmer.step3(stemsPart2);
		String stemsPart4 = stemmer.step4(stemsPart3);
		String stemsPart5 = stemmer.step5(stemsPart4);

		return stemsPart5;
	}

	public static Double geDocumentClassScoreByHypernym(
			HashMap<String, Integer> docTokens, boolean removeStopWord,
			HashMap<String, Integer> classVocab) {

		HashSet<String> docHypernyms = new HashSet<String>();
		HashSet<String> classHypernyms = new HashSet<String>();

		for (String token : docTokens.keySet()) {
			if (removeStopWord)
				if (NLPClassifier.checkStopWord(token))
					continue;
			String pos = wordnet.getBestPos(token);
			if (pos == null)
				pos = "a";
			String[] hypernyms = wordnet.getAllHypernyms(token, pos);
			if (hypernyms != null)
				for (int i = 0; i < hypernyms.length; i++) {
					docHypernyms.add(getStem(hypernyms[i]));
				}
		}

		for (String token : classVocab.keySet()) {
			if (removeStopWord)
				if (NLPClassifier.checkStopWord(token))
					continue;
			String pos = wordnet.getBestPos(token);
			if (pos == null)
				pos = "a";
			String[] hypernyms = wordnet.getAllHypernyms(token, pos);
			if (hypernyms != null)
				for (int i = 0; i < hypernyms.length; i++) {
					classHypernyms.add(getStem(hypernyms[i]));
				}
		}

		Integer size = classHypernyms.size();
		classHypernyms.retainAll(docHypernyms);
		Integer intersection = classHypernyms.size();
		Double score = intersection.doubleValue() / size;

		return score;
	}

	public static HashMap<NLPClassifier.DOCUMENT_CLASS, Double> getDocumentClass(
			HashMap<String, Integer> docTokens, boolean removeStopWord) {
		Double politicalScore = Hypernyms.geDocumentClassScoreByHypernym(
				docTokens, removeStopWord, NLPClassifier.politicalVocab);
		Double sportsScore = Hypernyms.geDocumentClassScoreByHypernym(
				docTokens, removeStopWord, NLPClassifier.sportsVocab);
		Double buisnessScore = Hypernyms.geDocumentClassScoreByHypernym(
				docTokens, removeStopWord, NLPClassifier.businessVocab);

		HashMap<NLPClassifier.DOCUMENT_CLASS, Double> scoreMap = new HashMap<NLPClassifier.DOCUMENT_CLASS, Double>();
		scoreMap.put(NLPClassifier.DOCUMENT_CLASS.POLITICAL, politicalScore);
		scoreMap.put(NLPClassifier.DOCUMENT_CLASS.BUSINESS, buisnessScore);
		scoreMap.put(NLPClassifier.DOCUMENT_CLASS.SPORTS, sportsScore);

		return scoreMap;
	}

}
