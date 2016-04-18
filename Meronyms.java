

import java.util.HashMap;
import java.util.HashSet;

import rita.RiWordNet;

public class Meronyms {
	public static RiWordNet wordnet = new RiWordNet(
			"thirdparty/WordNet/2.1/dict");

	public static Double geDocumentClassScoreByHypernym(
			HashMap<String, Integer> docTokens, boolean removeStopWord,
			HashMap<String, Integer> classVocab) {

		HashSet<String> docMeronyms = new HashSet<String>();
		HashSet<String> classMeronyms = new HashSet<String>();

		for (String token : docTokens.keySet()) {
			if (removeStopWord)
				if (NLPClassifier.checkStopWord(token))
					continue;
			String pos = wordnet.getBestPos(token);
			if (pos == null)
				pos = "a";
			String[] meronyms = wordnet.getAllMeronyms(token, pos);
			if (meronyms != null)
				for (int i = 0; i < meronyms.length; i++) {
					docMeronyms.add(meronyms[i]);
				}
		}

		for (String token : classVocab.keySet()) {
			if (removeStopWord)
				if (NLPClassifier.checkStopWord(token))
					continue;
			String pos = wordnet.getBestPos(token);
			if (pos == null)
				pos = "a";
			String[] meronyms = wordnet.getAllMeronyms(token, pos);
			if (meronyms != null)
				for (int i = 0; i < meronyms.length; i++) {
					classMeronyms.add(meronyms[i]);
				}
		}

		Integer size = classMeronyms.size();
		classMeronyms.retainAll(docMeronyms);
		Integer intersection = classMeronyms.size();
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
