import base64
import requests
from io import BytesIO
from PIL import Image
import torch
import pillow_heif
pillow_heif.register_heif_opener()
import cv2
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
import pytesseract
from typing import Tuple, Dict
import math
from collections import Counter
import re

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = torch.nn.Linear(model.last_channel, 1)
model.load_state_dict(torch.load('mobilenetv2_prescription_label.pth', map_location=device))
model.to(device).eval()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# OCR function
def ocr_via_checkrx(pil_img: Image.Image):
    try:
        pil_img.show()
        buf = BytesIO()
        pil_img.save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        resp = requests.post(
            'https://test.checkrx.com/api/v3/vision/',
            headers={'Accept': 'application/json'},
            json={
                "base64_image": img_b64
            },
            timeout=10
        )

        if resp.status_code == 200:
            print(resp.json())
            return resp.json()
        else:
            print(f"API Error: {resp.status_code} - {resp.text}")
            return None

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to OCR API - check URL and internet connection")
        return None
    except Exception as e:
        print(f"❌ OCR Error: {e}")
        return None

class OCRConfidenceScorer:
    def __init__(self):
        # Common OCR confusion patterns
        self.confusion_pairs = {
            ('0', 'O'), ('1', 'l'), ('1', 'I'), ('5', 'S'), ('6', 'G'),
            ('8', 'B'), ('rn', 'm'), ('cl', 'd'), ('vv', 'w'), ('nn', 'n')
        }

        # Common English letter frequencies (normalized)
        self.english_freq = {
            'e': 12.02, 't': 9.10, 'a': 8.12, 'o': 7.68, 'i': 6.97, 'n': 6.75,
            's': 6.33, 'h': 6.09, 'r': 5.99, 'd': 4.25, 'l': 4.03, 'c': 2.78,
            'u': 2.76, 'm': 2.41, 'w': 2.36, 'f': 2.23, 'g': 2.02, 'y': 1.97,
            'p': 1.93, 'b': 1.29, 'v': 0.98, 'k': 0.77, 'j': 0.15, 'x': 0.15,
            'q': 0.10, 'z': 0.07
        }

        # Common OCR artifacts
        self.ocr_artifacts = {
            '█', '▌', '▐', '║', '╣', '╠', '╦', '╩', '╬', '╧', '╨', '╤', '╥',
            '╙', '╘', '╒', '╓', '╫', '╪', '┘', '┌', '└', '┐', '─', '│', '┼',
            '□', '■', '●', '○', '◦', '•', '‣', '⁃', '∙', '▪', '▫', '◘', '◙',
            '♠', '♣', '♥', '♦', '☺', '☻', '☼', '♂', '♀', '♪', '♫', '☎', '☏'
        }

        # Problematic character sequences
        self.problematic_sequences = ['|||', '...', '---', '___', '^^^', '~~~']

        # Dictionary of common words (simplified - in practice, use a comprehensive dictionary)
        self.common_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who',
            'boy', 'did', 'man', 'end', 'few', 'got', 'let', 'put', 'say', 'she',
            'too', 'use', 'this', 'that', 'with', 'have', 'from', 'they', 'know',
            'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come',
            'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take',
            'than', 'them', 'well', 'were', 'will', 'would', 'there', 'could',
            'other', 'after', 'first', 'never', 'these', 'think', 'where', 'being',
            'every', 'great', 'might', 'shall', 'still', 'those', 'under', 'while',
            'before', 'should', 'through', 'people', 'little', 'right', 'world'
        }

    def calculate_confidence(self, text: str) -> Tuple[float, Dict]:
        """
        Calculate OCR confidence score (0-100) with detailed analysis
        """
        if not text or not text.strip():
            return 0.0, {"error": "Empty or null text"}

        text = text.strip()
        details = {}

        # Initialize weighted scoring components
        scores = {
            'character_quality': 0,
            'linguistic_coherence': 0,
            'structural_integrity': 0,
            'pattern_analysis': 0,
            'contextual_validation': 0
        }

        # Weights for each component (must sum to 1.0)
        weights = {
            'character_quality': 0.25,
            'linguistic_coherence': 0.30,
            'structural_integrity': 0.20,
            'pattern_analysis': 0.15,
            'contextual_validation': 0.10
        }

        # === 1. CHARACTER QUALITY ANALYSIS ===
        char_score, char_details = self._analyze_character_quality(text)
        scores['character_quality'] = char_score
        details['character_quality'] = char_details

        # === 2. LINGUISTIC COHERENCE ===
        ling_score, ling_details = self._analyze_linguistic_coherence(text)
        scores['linguistic_coherence'] = ling_score
        details['linguistic_coherence'] = ling_details

        # === 3. STRUCTURAL INTEGRITY ===
        struct_score, struct_details = self._analyze_structural_integrity(text)
        scores['structural_integrity'] = struct_score
        details['structural_integrity'] = struct_details

        # === 4. PATTERN ANALYSIS ===
        pattern_score, pattern_details = self._analyze_patterns(text)
        scores['pattern_analysis'] = pattern_score
        details['pattern_analysis'] = pattern_details

        # === 5. CONTEXTUAL VALIDATION ===
        context_score, context_details = self._analyze_contextual_validation(text)
        scores['contextual_validation'] = context_score
        details['contextual_validation'] = context_details

        # === FINAL SCORE CALCULATION ===
        final_score = sum(scores[component] * weights[component]
                         for component in scores)

        # Apply length-based confidence adjustment
        length_multiplier = self._calculate_length_multiplier(len(text))
        final_score *= length_multiplier

        # Ensure score is between 0-100
        final_score = max(0, min(100, final_score))

        details['component_scores'] = scores
        details['weights'] = weights
        details['length_multiplier'] = length_multiplier
        details['final_score'] = round(final_score, 1)

        return round(final_score, 1), details

    def _analyze_character_quality(self, text: str) -> Tuple[float, Dict]:
        """Analyze the quality of individual characters"""
        details = {}
        score = 100.0

        total_chars = len(text)
        if total_chars == 0:
            return 0, {"error": "No characters"}

        # Count various character types
        artifact_count = sum(1 for char in text if char in self.ocr_artifacts)
        unknown_count = text.count('?') + text.count('�') + text.count('□')
        control_count = sum(1 for char in text if ord(char) < 32 and char not in '\t\n\r')

        # Calculate ratios
        artifact_ratio = artifact_count / total_chars
        unknown_ratio = unknown_count / total_chars
        control_ratio = control_count / total_chars

        details.update({
            'artifact_count': artifact_count,
            'unknown_count': unknown_count,
            'control_count': control_count,
            'artifact_ratio': artifact_ratio,
            'unknown_ratio': unknown_ratio,
            'control_ratio': control_ratio
        })

        # Apply penalties using sigmoid function for smoother degradation
        score -= 80 * (1 - math.exp(-10 * artifact_ratio))
        score -= 70 * (1 - math.exp(-8 * unknown_ratio))
        score -= 60 * (1 - math.exp(-12 * control_ratio))

        # Character diversity bonus
        unique_chars = len(set(text.lower()))
        diversity_ratio = unique_chars / total_chars
        details['diversity_ratio'] = diversity_ratio

        if diversity_ratio > 0.4:
            score += 15 * min(1, diversity_ratio)

        return max(0, score), details

    def _analyze_linguistic_coherence(self, text: str) -> Tuple[float, Dict]:
        """Analyze linguistic patterns and coherence"""
        details = {}
        score = 100.0

        # Letter frequency analysis
        letter_freq = Counter(char.lower() for char in text if char.isalpha())
        total_letters = sum(letter_freq.values())

        if total_letters == 0:
            return 50.0, {"error": "No letters found"}

        # Calculate chi-square statistic for letter frequency
        chi_square = 0
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            observed = letter_freq.get(letter, 0)
            expected = (self.english_freq.get(letter, 0) / 100) * total_letters
            if expected > 0:
                chi_square += ((observed - expected) ** 2) / expected

        # Normalize chi-square (lower is better)
        normalized_chi = min(100, chi_square / total_letters * 100)
        frequency_score = 100 - normalized_chi

        details['letter_frequency_score'] = frequency_score
        details['chi_square'] = chi_square

        # Bigram and trigram analysis
        bigram_score = self._analyze_ngrams(text, 2)
        trigram_score = self._analyze_ngrams(text, 3)

        details['bigram_score'] = bigram_score
        details['trigram_score'] = trigram_score

        # Combine linguistic scores
        score = (frequency_score * 0.4 + bigram_score * 0.35 + trigram_score * 0.25)

        return max(0, score), details

    def _analyze_ngrams(self, text: str, n: int) -> float:
        """Analyze n-gram patterns for naturalness"""
        # Common English n-grams (simplified)
        common_bigrams = {
            'th', 'he', 'in', 'er', 'an', 're', 'ed', 'nd', 'on', 'en',
            'at', 'ou', 'it', 'is', 'or', 'ti', 'hi', 'st', 'et', 'ng'
        }

        common_trigrams = {
            'the', 'and', 'ing', 'her', 'hat', 'his', 'tha', 'ere', 'for',
            'ent', 'ion', 'ter', 'was', 'you', 'ith', 'ver', 'all', 'wit'
        }

        if n == 2:
            common_ngrams = common_bigrams
        elif n == 3:
            common_ngrams = common_trigrams
        else:
            return 50.0

        # Extract n-grams
        ngrams = []
        clean_text = re.sub(r'[^a-zA-Z]', '', text.lower())

        for i in range(len(clean_text) - n + 1):
            ngrams.append(clean_text[i:i+n])

        if not ngrams:
            return 50.0

        # Calculate common n-gram ratio
        common_count = sum(1 for ngram in ngrams if ngram in common_ngrams)
        common_ratio = common_count / len(ngrams)

        # Score based on common n-gram frequency
        return min(100, common_ratio * 150)

    def _analyze_structural_integrity(self, text: str) -> Tuple[float, Dict]:
        """Analyze text structure and formatting"""
        details = {}
        score = 100.0

        # Word analysis
        words = text.split()
        details['word_count'] = len(words)

        if not words:
            return 0, {"error": "No words found"}

        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        details['avg_word_length'] = avg_word_length

        # Penalize extreme word lengths
        if avg_word_length < 2:
            score -= 30
        elif avg_word_length > 12:
            score -= 20

        # Check for valid word patterns
        valid_words = sum(1 for word in words if re.match(r'^[a-zA-Z]+$', word))
        valid_ratio = valid_words / len(words)
        details['valid_word_ratio'] = valid_ratio

        score = score * (0.3 + 0.7 * valid_ratio)

        # Spacing analysis
        space_count = text.count(' ')
        total_chars = len(text)
        space_ratio = space_count / total_chars if total_chars > 0 else 0
        details['space_ratio'] = space_ratio

        # Optimal spacing is around 15-20% of characters
        if 0.12 <= space_ratio <= 0.25:
            score += 10
        elif space_ratio < 0.05 or space_ratio > 0.4:
            score -= 15

        # Check for problematic sequences
        problematic_count = sum(text.count(seq) for seq in self.problematic_sequences)
        if problematic_count > 0:
            score -= min(30, problematic_count * 10)

        details['problematic_sequences'] = problematic_count

        return max(0, score), details

    def _analyze_patterns(self, text: str) -> Tuple[float, Dict]:
        """Analyze OCR-specific patterns and artifacts"""
        details = {}
        score = 100.0

        # Check for OCR confusion patterns
        confusion_count = 0
        for pair in self.confusion_pairs:
            confusion_count += text.count(pair[0] + pair[1])
            confusion_count += text.count(pair[1] + pair[0])

        details['confusion_patterns'] = confusion_count

        if confusion_count > 0:
            score -= min(25, confusion_count * 5)

        # Check for repeated characters (OCR artifacts)
        repeated_pattern = re.findall(r'(.)\1{2,}', text)
        repeated_count = len(repeated_pattern)
        details['repeated_sequences'] = repeated_count

        if repeated_count > 0:
            score -= min(20, repeated_count * 8)

        # Check case consistency
        if text.isupper() and len(text) > 20:
            score -= 15
        elif text.islower() and len(text) > 20:
            score -= 10

        # Mixed case bonus (natural text usually has mixed case)
        has_upper = any(c.isupper() for c in text)
        has_lower = any(c.islower() for c in text)
        if has_upper and has_lower:
            score += 10

        details['case_analysis'] = {
            'has_upper': has_upper,
            'has_lower': has_lower,
            'is_all_upper': text.isupper(),
            'is_all_lower': text.islower()
        }

        return max(0, score), details

    def _analyze_contextual_validation(self, text: str) -> Tuple[float, Dict]:
        """Validate text against contextual knowledge"""
        details = {}
        score = 100.0

        words = [word.lower().strip('.,!?;:') for word in text.split()]

        if not words:
            return 50, {"error": "No words to validate"}

        # Dictionary validation
        valid_words = sum(1 for word in words if word in self.common_words)
        dict_ratio = valid_words / len(words)
        details['dictionary_ratio'] = dict_ratio

        # Score based on dictionary word ratio
        score = 30 + (dict_ratio * 70)

        # Check for numeric patterns that might be dates, times, etc.
        numeric_patterns = len(re.findall(r'\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b', text))
        time_patterns = len(re.findall(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', text))

        if numeric_patterns > 0 or time_patterns > 0:
            score += 5  # Bonus for structured numeric data

        details['numeric_patterns'] = numeric_patterns
        details['time_patterns'] = time_patterns

        return max(0, score), details

    def _calculate_length_multiplier(self, length: int) -> float:
        """Calculate confidence multiplier based on text length"""
        if length < 3:
            return 0.3
        elif length < 10:
            return 0.6 + (length - 3) * 0.05
        elif length < 50:
            return 0.9 + (length - 10) * 0.0025
        elif length < 200:
            return 1.0
        else:
            return max(0.95, 1.0 - (length - 200) * 0.0001)

# Usage example
def calculate_ocr_confidence(text: str) -> Tuple[float, Dict]:
    """
    Main function to calculate OCR confidence score
    """
    scorer = OCRConfidenceScorer()
    return scorer.calculate_confidence(text)
    """OCR with confidence scoring to handle blurry text"""
    result = ocr_via_checkrx(pil_img)

    if result is None:
        return None, 0, {}

    # Extract text from result
    if isinstance(result, dict):
        text = result.get('text', '') or result.get('result', '') or str(result)
    else:
        text = str(result)

    # Calculate confidence
    confidence_score, details = calculate_ocr_confidence(text)

    # Print detailed analysis
    print(f"=== OCR CONFIDENCE ANALYSIS ===")
    print(f"Text: {text}")
    print(f"Confidence Score: {confidence_score}/100")
    print(f"Details: {details}")
    print(f"==============================\n")

    # Return only 2 values like before
    return text, confidence_score  # Remove the details



TH_LOW, TH_HIGH = 0.95, 1.00

cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(img_t)
        prob_no_label = torch.sigmoid(logit).item()
        prob_label = 1 - prob_no_label

    label = f"Prescription: {'YES' if prob_label > 0.5 else 'NO'} ({prob_label:.2f})"
    color = (0, 255, 0) if prob_label > 0.5 else (0, 0, 255)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Prescription Detector', frame)

    # Only call OCR in your threshold range
    if TH_LOW <= prob_label <= TH_HIGH:
        print(f"In range block: {prob_label}\n")
        ocr_text = pytesseract.image_to_string(img)
        confidence, _ = calculate_ocr_confidence(ocr_text)
        print(f"OCR text: {ocr_text}")
        print(f"OCR confidence: {confidence}")

        if confidence > 65:
            print("Sending to API!")
            buf = BytesIO()
            img.save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            try:
                resp = requests.post(
                    'https://test.checkrx.com/api/v3/vision/',
                    headers={'Accept': 'application/json'},
                    json={"base64_image": img_b64},
                    timeout=10
                )
                if resp.status_code == 200:
                    print("API response:", resp.json())
                else:
                    print(f"API Error: {resp.status_code} - {resp.text}")
            except Exception as e:
                print(f"API call failed: {e}")
        else:
            print("Not sending to API: OCR confidence too low.")


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
