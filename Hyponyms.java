

import java.util.HashMap;
import java.util.HashSet;

import rita.RiWordNet;

public class Hyponyms {
	public static RiWordNet wordnet = new RiWordNet(
			"thirdparty/WordNet/2.1/dict");

	public static Double geDocumentClassScoreByHypernym(
			HashMap<String, Integer> docTokens, boolean removeStopWord,
			HashMap<String, Integer> classVocab) {

		HashSet<String> docHyponyms = new HashSet<String>();
		HashSet<String> classHyponyms = new HashSet<String>();

		for (String token : docTokens.keySet()) {
			if (removeStopWord)
				if (NLPClassifier.checkStopWord(token))
					continue;
			String pos = wordnet.getBestPos(token);
			if (pos == null)
				pos = "a";
			String[] hyponyms = wordnet.getAllHolonyms(token, pos);
			if (hyponyms != null)
				for (int i = 0; i < hyponyms.length; i++) {
					docHyponyms.add(hyponyms[i]);
				}
		}

		for (String token : classVocab.keySet()) {
			if (removeStopWord)
				if (NLPClassifier.checkStopWord(token))
					continue;
			String pos = wordnet.getBestPos(token);
			if (pos == null)
				pos = "a";
			String[] hyponyms = wordnet.getAllHolonyms(token, pos);
			if (hyponyms != null)
				for (int i = 0; i < hyponyms.length; i++) {
					classHyponyms.add(hyponyms[i]);
				}
		}

		Integer size = classHyponyms.size();
		classHyponyms.retainAll(docHyponyms);
		Integer intersection = classHyponyms.size();
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
