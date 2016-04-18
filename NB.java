
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

public class NB {
	public static int totalPoliticalCount = 0;
	public static int totalSportsCount = 0;
	public static int totalBusinessCount = 0;
	public static int politicalDocCount = 0;
	public static int sportsDocCount = 0;
	public static int businessDocCount = 0;

	public static HashMap<String, Integer> politicalVocab = new HashMap<String, Integer>();
	public static HashMap<String, Double> politicalProbablity = new HashMap<String, Double>();
	public static HashMap<String, Integer> sportsVocab = new HashMap<String, Integer>();
	public static HashMap<String, Double> sportsProbablity = new HashMap<String, Double>();
	public static HashMap<String, Integer> businessVocab = new HashMap<String, Integer>();
	public static HashMap<String, Double> businessProbablity = new HashMap<String, Double>();
	public static Set<String> vocabSet = new HashSet<String>();
	public static String splitter = "\\s+";

	public static enum DOCUMENT_CLASS {
		POLITICAL, SPORTS, BUSINESS
	}

	public static void main(String[] args) {

		String learnPoliticalSrc = "corpus/poltics_train";// args[0];
		String learnSportsSrc = "corpus/sports_train";// args[1];
		String learnBuisnessSrc = "corpus/buisness_train";// args[1];
		String testPoliticalSrcFile = "corpus/politics_test";// args[2];
		String testSportsSrcFile = "corpus/sports_test";// args[3];
		String testBuisnessSrcFile = "corpus/buisness_test";

		File testPoliticalSrc = new File(testPoliticalSrcFile);
		File testSportsSrc = new File(testSportsSrcFile);
		File testBuisnessSrc = new File(testBuisnessSrcFile);
		File politicalFolder = new File(learnPoliticalSrc);
		File sportsFolder = new File(learnSportsSrc);
		File buisnessFolder = new File(learnBuisnessSrc);

		try {
			// Naive bayes
			learnFromPolictical(politicalFolder);
			learnFromSports(sportsFolder);
			learnFromBuisness(buisnessFolder);
			vocabSet.addAll(politicalVocab.keySet());
			vocabSet.addAll(sportsVocab.keySet());
			vocabSet.addAll(businessVocab.keySet());
			politicalProbablity();
			sportsProbablity();
			buisnessProbablity();
			testPolitical(testPoliticalSrc);
			testSports(testSportsSrc);
			testBuisness(testBuisnessSrc);

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public static void politicalProbablity() {
		Set<String> vocab = politicalVocab.keySet();
		Integer denominator = totalPoliticalCount
				+ politicalVocab.keySet().size();

		for (String token : vocab) {
			Integer wordCount = politicalVocab.get(token);
			Double probablity = wordCount / Double.valueOf(denominator);
			politicalProbablity.put(token, probablity);
		}

	}

	public static void sportsProbablity() {
		Set<String> vocab = sportsVocab.keySet();
		Integer denominator = totalSportsCount + sportsVocab.keySet().size();

		for (String token : vocab) {
			Integer wordCount = sportsVocab.get(token);
			Double probablity = wordCount / Double.valueOf(denominator);
			sportsProbablity.put(token, probablity);
		}

	}

	public static void buisnessProbablity() {
		Set<String> vocab = businessVocab.keySet();
		Integer denominator = totalBusinessCount
				+ businessVocab.keySet().size();

		for (String token : vocab) {
			Integer wordCount = businessVocab.get(token);
			Double probablity = wordCount / Double.valueOf(denominator);
			businessProbablity.put(token, probablity);
		}

	}

	

	public static DOCUMENT_CLASS getDocumentClass(
			HashMap<String, Integer> docTokens, boolean removeStopWord) {
		Double politicalScore;
		Double sportsScore;
		Double buisnessScore;

		Double politicalDocProbablity = politicalDocCount
				/ Double.valueOf(politicalDocCount + sportsDocCount
						+ businessDocCount);
		Double sportsDocProbablity = sportsDocCount
				/ Double.valueOf(politicalDocCount + sportsDocCount
						+ businessDocCount);
		Double buisnessDocProbablity = businessDocCount
				/ Double.valueOf(politicalDocCount + sportsDocCount
						+ businessDocCount);

		politicalScore = Math.log(politicalDocProbablity) / Math.log(2);
		sportsScore = Math.log(sportsDocProbablity) / Math.log(2);
		buisnessScore = Math.log(buisnessDocProbablity) / Math.log(2);
		for (String token : docTokens.keySet()) {
			if (removeStopWord)
				if (checkStopWord(token))
					continue;

			int tokenCount = docTokens.get(token);

			while (tokenCount > 0) {
				if (politicalVocab.containsKey(token)) {
					int politicalVocabTokenCount = docTokens.get(token);
					while (politicalVocabTokenCount != 0) {
						Double prob = politicalProbablity.get(token);
						politicalScore = politicalScore + Math.log(prob)
								/ Math.log(2);
						politicalVocabTokenCount--;
					}
				} else {
					int politicalVocabTokenCount = docTokens.get(token);
					while (politicalVocabTokenCount != 0) {
						Double x = 1 / Double.valueOf(totalPoliticalCount
								+ vocabSet.size());
						politicalScore = politicalScore + Math.log(x)
								/ Math.log(2);
						politicalVocabTokenCount--;
					}
				}

				tokenCount--;
			}
		}
		for (String token : docTokens.keySet()) {

			if (removeStopWord)
				if (checkStopWord(token))
					continue;
			int tokenCount = docTokens.get(token);

			while (tokenCount > 0) {

				if (sportsVocab.containsKey(token)) {
					int sportsVocabTokenCount = docTokens.get(token);
					while (sportsVocabTokenCount != 0) {
						sportsScore = sportsScore
								+ Math.log(sportsProbablity.get(token))
								/ Math.log(2);
						sportsVocabTokenCount--;
					}
				} else {
					int sportsVocabTokenCount = docTokens.get(token);
					while (sportsVocabTokenCount != 0) {
						Double x = 1 / Double.valueOf(totalSportsCount
								+ vocabSet.size());
						sportsScore = sportsScore + Math.log(x) / Math.log(2);
						sportsVocabTokenCount--;
					}
				}
				tokenCount--;
			}

		}

		for (String token : docTokens.keySet()) {

			if (removeStopWord)
				if (checkStopWord(token))
					continue;
			int tokenCount = docTokens.get(token);

			while (tokenCount > 0) {

				if (businessVocab.containsKey(token)) {
					int buisnessVocabTokenCount = docTokens.get(token);
					while (buisnessVocabTokenCount != 0) {
						buisnessScore = buisnessScore
								+ Math.log(businessProbablity.get(token))
								/ Math.log(2);
						buisnessVocabTokenCount--;
					}
				} else {
					int buisnessVocabTokenCount = docTokens.get(token);
					while (buisnessVocabTokenCount != 0) {
						Double x = 1 / Double.valueOf(totalBusinessCount
								+ vocabSet.size());
						buisnessScore = buisnessScore + Math.log(x)
								/ Math.log(2);
						buisnessVocabTokenCount--;
					}
				}
				tokenCount--;
			}

		}

		if (politicalScore > sportsScore && politicalScore > buisnessScore)
			return DOCUMENT_CLASS.POLITICAL;
		else if (sportsScore > politicalScore && sportsScore > buisnessScore)
			return DOCUMENT_CLASS.SPORTS;
		else
			return DOCUMENT_CLASS.BUSINESS;

	}

	public static void testSports(File spamFolder)
			throws FileNotFoundException, IOException {

		int docCount = 0;
		int correctCountWithoutStopWords = 0;
		int correctCountWithStopWords = 0;
		String files[] = spamFolder.list();
		for (String file : files) {
			docCount++;
			File newFile = new File(spamFolder, file);
			BufferedReader reader = new BufferedReader(new FileReader(newFile));
			HashMap<String, Integer> docVocab = new HashMap<String, Integer>();

			String line;
			while ((line = reader.readLine()) != null) {
				line = line.replace("-", "");
				line = line.replace(",", "");
				line = line.replace(" ,", "");
				line = line.replace(", ", "");
				line = line.replace(".", "");
				line = line.replace(":", "");
				line = line.replace(" : ", "");
				line = line.replace(" / ", "");
				Set<String> existingTokens = docVocab.keySet();
				String[] tokens = line.split(splitter);
				for (int i = 0; i < tokens.length; i++) {
					if (tokens[i].equals(""))
						continue;

					if (existingTokens.contains(tokens[i])) {
						Integer count = docVocab.get(tokens[i]);
						count++;
						docVocab.put(tokens[i], count);
					} else {
						docVocab.put(tokens[i], 1);
					}

				}

			}

			if (DOCUMENT_CLASS.SPORTS.equals(getDocumentClass(docVocab, false))) {
				correctCountWithStopWords++;
			}
			if (DOCUMENT_CLASS.SPORTS.equals(getDocumentClass(docVocab, true))) {
				correctCountWithoutStopWords++;
			}
			reader.close();

		}

		System.out.println("Naive Bayes Sports Accuracy With Stop Word: "
				+ correctCountWithStopWords / Double.valueOf(docCount) * 100);
		System.out
				.println("Naive Bayes Sports Accuracy Without Stop Word: "
						+ correctCountWithoutStopWords
						/ Double.valueOf(docCount) * 100);

	}

	public static void testBuisness(File buisnessFolder)
			throws FileNotFoundException, IOException {

		int docCount = 0;
		int correctCountWithoutStopWords = 0;
		int correctCountWithStopWords = 0;
		String files[] = buisnessFolder.list();
		for (String file : files) {
			docCount++;
			File newFile = new File(buisnessFolder, file);
			BufferedReader reader = new BufferedReader(new FileReader(newFile));
			HashMap<String, Integer> docVocab = new HashMap<String, Integer>();

			String line;
			while ((line = reader.readLine()) != null) {
				line = line.replace("-", "");
				line = line.replace(",", "");
				line = line.replace(" ,", "");
				line = line.replace(", ", "");
				line = line.replace(".", "");
				line = line.replace(":", "");
				line = line.replace(" : ", "");
				line = line.replace(" / ", "");
				Set<String> existingTokens = docVocab.keySet();
				String[] tokens = line.split(splitter);
				for (int i = 0; i < tokens.length; i++) {
					if (tokens[i].equals(""))
						continue;

					if (existingTokens.contains(tokens[i])) {
						Integer count = docVocab.get(tokens[i]);
						count++;
						docVocab.put(tokens[i], count);
					} else {
						docVocab.put(tokens[i], 1);
					}

				}

			}

			if (DOCUMENT_CLASS.BUSINESS
					.equals(getDocumentClass(docVocab, false))) {
				correctCountWithStopWords++;
			}
			if (DOCUMENT_CLASS.BUSINESS
					.equals(getDocumentClass(docVocab, true))) {
				correctCountWithoutStopWords++;
			}

			reader.close();

		}

		System.out
				.println("Naive Bayes Buisness Corpus Accuracy With Stop Word: "
						+ correctCountWithStopWords
						/ Double.valueOf(docCount)
						* 100);
		System.out
				.println("Naive Bayes Buisness Corpus Accuracy Without Stop Word: "
						+ correctCountWithoutStopWords
						/ Double.valueOf(docCount) * 100);

	}

	public static void testPolitical(File politicalFolder)
			throws FileNotFoundException, IOException {

		int docCount = 0;
		int correctCountWithStopWords = 0;
		int correctCountWithoutStopWords = 0;
		String files[] = politicalFolder.list();
		for (String file : files) {
			docCount++;
			File newFile = new File(politicalFolder, file);
			BufferedReader reader = new BufferedReader(new FileReader(newFile));
			HashMap<String, Integer> docVocab = new HashMap<String, Integer>();

			String line;
			while ((line = reader.readLine()) != null) {
				line = line.replace("-", "");
				line = line.replace(",", "");
				line = line.replace(" ,", "");
				line = line.replace(", ", "");
				line = line.replace(".", "");
				line = line.replace(":", "");
				line = line.replace(" : ", "");
				line = line.replace(" / ", "");
				Set<String> existingTokens = docVocab.keySet();
				String[] tokens = line.split(splitter);
				for (int i = 0; i < tokens.length; i++) {
					if (tokens[i].equals(""))
						continue;

					if (existingTokens.contains(tokens[i])) {
						Integer count = docVocab.get(tokens[i]);
						count++;
						docVocab.put(tokens[i], count);
					} else {
						docVocab.put(tokens[i], 1);
					}

				}

			}

			if (DOCUMENT_CLASS.POLITICAL.equals(getDocumentClass(docVocab,
					false))) {
				correctCountWithStopWords++;
			}
			if (DOCUMENT_CLASS.POLITICAL
					.equals(getDocumentClass(docVocab, true))) {
				correctCountWithoutStopWords++;
			}

			reader.close();

		}
		System.out.println("Naive Bayes Political Accuracy With Stop Word: "
				+ correctCountWithStopWords / Double.valueOf(docCount) * 100);
		System.out
				.println("Naive Bayes Political Accuracy Without Stop Word: "
						+ correctCountWithoutStopWords
						/ Double.valueOf(docCount) * 100);

	}

	public static void learnFromPolictical(File politicalFolder)
			throws IOException {
		// list all the files in directory
		String files[] = politicalFolder.list();
		for (String file : files) {
			politicalDocCount++;
			File newFile = new File(politicalFolder, file);
			BufferedReader reader = new BufferedReader(new FileReader(newFile));
			String line;
			while ((line = reader.readLine()) != null) {
				line = line.replace("-", "");
				line = line.replace(",", "");
				line = line.replace(" ,", "");
				line = line.replace(", ", "");
				line = line.replace(".", "");
				line = line.replace(":", "");
				line = line.replace(" : ", "");
				line = line.replace(" / ", "");
				Set<String> existingTokens = politicalVocab.keySet();
				String[] tokens = line.split(splitter);
				for (int i = 0; i < tokens.length; i++) {
					if (tokens[i].equals(""))
						continue;

					totalPoliticalCount++;
					if (existingTokens.contains(tokens[i])) {
						Integer count = politicalVocab.get(tokens[i]);
						count++;
						politicalVocab.put(tokens[i], count);
					} else {
						politicalVocab.put(tokens[i], 1);
					}

				}

			}
			reader.close();
		}

	}

	public static void learnFromSports(File sportsFolder) throws IOException {
		// list all the files in directory
		String files[] = sportsFolder.list();
		for (String file : files) {
			sportsDocCount++;
			File newFile = new File(sportsFolder, file);
			BufferedReader reader = new BufferedReader(new FileReader(newFile));
			String line;
			while ((line = reader.readLine()) != null) {

				line = line.replace("-", "");
				line = line.replace(",", "");
				line = line.replace(".", "");
				line = line.replace("/", "");
				line = line.replace(":", "");
				Set<String> existingTokens = sportsVocab.keySet();
				String[] tokens = line.split(splitter);
				for (int i = 0; i < tokens.length; i++) {
					if (tokens[i].equals(""))
						continue;

					totalSportsCount++;
					if (existingTokens.contains(tokens[i])) {
						Integer count = sportsVocab.get(tokens[i]);
						count++;
						sportsVocab.put(tokens[i], count);
					} else {
						sportsVocab.put(tokens[i], 1);
					}

				}

			}
			reader.close();
		}
	}

	public static void learnFromBuisness(File buisnessFolder)
			throws IOException {
		// list all the files in directory
		String files[] = buisnessFolder.list();
		for (String file : files) {
			businessDocCount++;
			File newFile = new File(buisnessFolder, file);
			BufferedReader reader = new BufferedReader(new FileReader(newFile));
			String line;
			while ((line = reader.readLine()) != null) {

				line = line.replace("-", "");
				line = line.replace(",", "");
				line = line.replace(".", "");
				line = line.replace("/", "");
				line = line.replace(":", "");
				Set<String> existingTokens = businessVocab.keySet();
				String[] tokens = line.split(splitter);
				for (int i = 0; i < tokens.length; i++) {
					if (tokens[i].equals(""))
						continue;

					totalBusinessCount++;
					if (existingTokens.contains(tokens[i])) {
						Integer count = businessVocab.get(tokens[i]);
						count++;
						businessVocab.put(tokens[i], count);
					} else {
						businessVocab.put(tokens[i], 1);
					}

				}

			}
			reader.close();
		}
	}

	public static boolean checkStopWord(String token) {

		String stopWord = "a about above after again against all am an and any are aren't as "
				+ "at be because been before being below between both but by can't cannot could "
				+ "couldn't did didn't do does doesn't doing don't down during each few for from "
				+ "further had hadn't has hasn't have haven't having he he'd he'll he's her here here's "
				+ "hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its "
				+ "itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours	 "
				+ "ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the "
				+ "their theirs them themselves then there there's these they they'd they'll they're they've this those through to too "
				+ "under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which "
				+ "while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves";

		String[] stopWordArray = stopWord.split(" ");
		for (int i = 0; i < stopWordArray.length; i++) {
			if (token.equals(stopWordArray[i]))
				return true;
		}

		return false;

	}
}
