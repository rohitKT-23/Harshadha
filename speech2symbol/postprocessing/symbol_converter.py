"""
Advanced post-processing module for converting spoken operators to symbols
Includes context-aware rules and NLP-based disambiguation
"""

import re
import spacy
import nltk
from typing import Dict, List, Tuple, Optional, Set
import logging
from dataclasses import dataclass
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversionRule:
    """Rule for operator conversion"""
    pattern: str
    replacement: str
    context_required: Optional[List[str]] = None
    context_forbidden: Optional[List[str]] = None
    priority: int = 0
    case_sensitive: bool = False

class ContextAwareSymbolConverter:
    """Advanced symbol converter with contextual understanding"""
    
    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy
        
        # Initialize NLP model
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except IOError:
                logger.warning("spaCy model not found. Falling back to rule-based approach.")
                self.nlp = None
                self.use_spacy = False
        
        # Initialize NLTK components
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('wordnet', quiet=True)
        except:
            logger.warning("NLTK data not available")
        
        # Define conversion rules with context awareness
        self.conversion_rules = self._create_conversion_rules()
        
        # Mathematical context indicators
        self.math_contexts = {
            'equation_indicators': ['equals', 'equal', 'is', 'makes', 'gives'],
            'calculation_words': ['calculate', 'compute', 'solve', 'find', 'determine'],
            'number_words': ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'],
            'units': ['percent', 'percentage', 'degrees', 'dollars', 'cents']
        }
        
        # Punctuation context indicators
        self.punctuation_contexts = {
            'sentence_end': ['end', 'finish', 'close', 'conclude'],
            'list_context': ['first', 'second', 'third', 'next', 'then', 'finally'],
            'question_context': ['ask', 'question', 'inquire', 'wonder'],
            'emphasis_context': ['important', 'urgent', 'critical', 'wow', 'amazing']
        }
    
    def _create_conversion_rules(self) -> List[ConversionRule]:
        """Create comprehensive conversion rules with priorities"""
        rules = []
        
        # Mathematical operators (high priority)
        math_rules = [
            ConversionRule(r'\bplus\b', '+', ['number', 'digit'], priority=10),
            ConversionRule(r'\badd(?:ed)?\s+to\b', '+', ['number'], priority=9),
            ConversionRule(r'\bminus\b', '-', ['number', 'digit'], priority=10),
            ConversionRule(r'\bsubtract(?:ed)?\s+(?:from)?\b', '-', ['number'], priority=9),
            ConversionRule(r'\btimes\b', '×', ['number', 'digit'], priority=10),
            ConversionRule(r'\bmultipli(?:ed|es?)\s+by\b', '×', ['number'], priority=9),
            ConversionRule(r'\bdivide[ds]?\s+by\b', '÷', ['number'], priority=9),
            ConversionRule(r'\bover\b', '/', ['number', 'fraction'], priority=8),
            ConversionRule(r'\bequals?\b', '=', ['number', 'result'], priority=10),
            ConversionRule(r'\bis\s+equal\s+to\b', '=', ['number'], priority=9),
            ConversionRule(r'\bgreater\s+than\b', '>', ['number', 'comparison'], priority=9),
            ConversionRule(r'\bless\s+than\b', '<', ['number', 'comparison'], priority=9),
            ConversionRule(r'\bgreater\s+than\s+or\s+equal\s+to\b', '≥', ['number'], priority=8),
            ConversionRule(r'\bless\s+than\s+or\s+equal\s+to\b', '≤', ['number'], priority=8),
            ConversionRule(r'\bpercent\b', '%', ['number', 'rate'], priority=10),
            ConversionRule(r'\bpercentage\b', '%', ['number', 'rate'], priority=9),
        ]
        
        # Punctuation rules (medium priority)
        punctuation_rules = [
            ConversionRule(r'\bcomma\b', ',', ['list', 'pause'], priority=7),
            ConversionRule(r'\bperiod\b', '.', ['end', 'sentence'], priority=8),
            ConversionRule(r'\bdot\b', '.', ['decimal', 'abbreviation'], priority=6),
            ConversionRule(r'\bquestion\s+mark\b', '?', ['question', 'ask'], priority=8),
            ConversionRule(r'\bexclamation\s+(?:mark|point)\b', '!', ['emphasis', 'surprise'], priority=8),
            ConversionRule(r'\bsemicolon\b', ';', ['list', 'clause'], priority=7),
            ConversionRule(r'\bcolon\b', ':', ['list', 'explanation'], priority=7),
            ConversionRule(r'\bapostrophe\b', "'", ['possession', 'contraction'], priority=6),
            ConversionRule(r'\bquote\b', '"', ['quotation', 'speech'], priority=6),
            ConversionRule(r'\bquotation\s+mark\b', '"', ['quotation'], priority=7),
        ]
        
        # Parentheses and brackets (lower priority due to ambiguity)
        bracket_rules = [
            ConversionRule(r'\bleft\s+parenthesis\b', '(', ['grouping'], priority=5),
            ConversionRule(r'\bright\s+parenthesis\b', ')', ['grouping'], priority=5),
            ConversionRule(r'\bopen\s+(?:paren|parenthesis)\b', '(', ['grouping'], priority=4),
            ConversionRule(r'\bclose\s+(?:paren|parenthesis)\b', ')', ['grouping'], priority=4),
            ConversionRule(r'\bleft\s+bracket\b', '[', ['array', 'index'], priority=5),
            ConversionRule(r'\bright\s+bracket\b', ']', ['array', 'index'], priority=5),
            ConversionRule(r'\bleft\s+brace\b', '{', ['set', 'code'], priority=4),
            ConversionRule(r'\bright\s+brace\b', '}', ['set', 'code'], priority=4),
        ]
        
        # Currency and symbols
        symbol_rules = [
            ConversionRule(r'\bdollar(?:s)?\b', '$', ['money', 'cost'], priority=9),
            ConversionRule(r'\bcents?\b', '¢', ['money', 'small'], priority=8),
            ConversionRule(r'\bat\s+sign\b', '@', ['email', 'location'], priority=8),
            ConversionRule(r'\bhashtag\b', '#', ['social', 'number'], priority=7),
            ConversionRule(r'\bampersand\b', '&', ['and', 'company'], priority=6),
            ConversionRule(r'\basterisk\b', '*', ['multiply', 'footnote'], priority=6),
            ConversionRule(r'\bunderscore\b', '_', ['space', 'code'], priority=5),
        ]
        
        rules.extend(math_rules)
        rules.extend(punctuation_rules)
        rules.extend(bracket_rules)
        rules.extend(symbol_rules)
        
        # Sort by priority (highest first)
        rules.sort(key=lambda x: x.priority, reverse=True)
        
        return rules
    
    def convert_text(self, text: str, confidence_threshold: float = 0.7) -> Tuple[str, Dict]:
        """Convert spoken operators to symbols with confidence scoring"""
        original_text = text
        converted_text = text
        conversion_log = []
        
        # Analyze context if spaCy is available
        context_analysis = self._analyze_context(text) if self.use_spacy else {}
        
        # Apply conversion rules
        for rule in self.conversion_rules:
            matches = list(re.finditer(rule.pattern, converted_text, 
                                     re.IGNORECASE if not rule.case_sensitive else 0))
            
            for match in reversed(matches):  # Reverse to maintain indices
                start, end = match.span()
                matched_text = match.group()
                
                # Check context requirements
                confidence = self._calculate_confidence(
                    matched_text, rule, converted_text, start, end, context_analysis
                )
                
                if confidence >= confidence_threshold:
                    # Apply conversion
                    converted_text = (converted_text[:start] + 
                                    rule.replacement + 
                                    converted_text[end:])
                    
                    conversion_log.append({
                        'original': matched_text,
                        'converted': rule.replacement,
                        'position': start,
                        'confidence': confidence,
                        'rule_priority': rule.priority
                    })
        
        # Post-process for common issues
        converted_text = self._post_process_corrections(converted_text)
        
        # Return results with metadata
        return converted_text, {
            'original_text': original_text,
            'conversions': conversion_log,
            'context_analysis': context_analysis,
            'total_conversions': len(conversion_log)
        }
    
    def _analyze_context(self, text: str) -> Dict:
        """Analyze text context using NLP"""
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        
        analysis = {
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'pos_tags': [(token.text, token.pos_) for token in doc],
            'has_numbers': any(token.like_num for token in doc),
            'has_email': any('@' in token.text for token in doc),
            'sentence_count': len(list(doc.sents)),
            'mathematical_context': self._detect_mathematical_context(doc),
            'punctuation_context': self._detect_punctuation_context(doc)
        }
        
        return analysis
    
    def _detect_mathematical_context(self, doc) -> Dict:
        """Detect mathematical context in text"""
        context = {
            'has_equation': False,
            'has_calculation': False,
            'has_numbers': False,
            'has_units': False
        }
        
        text_lower = doc.text.lower()
        
        # Check for equation indicators
        for indicator in self.math_contexts['equation_indicators']:
            if indicator in text_lower:
                context['has_equation'] = True
                break
        
        # Check for calculation words
        for word in self.math_contexts['calculation_words']:
            if word in text_lower:
                context['has_calculation'] = True
                break
        
        # Check for numbers
        context['has_numbers'] = any(token.like_num for token in doc)
        
        # Check for units
        for unit in self.math_contexts['units']:
            if unit in text_lower:
                context['has_units'] = True
                break
        
        return context
    
    def _detect_punctuation_context(self, doc) -> Dict:
        """Detect punctuation context in text"""
        context = {
            'has_list': False,
            'has_question': False,
            'has_emphasis': False,
            'sentence_end': False
        }
        
        text_lower = doc.text.lower()
        
        # Check for list context
        for indicator in self.punctuation_contexts['list_context']:
            if indicator in text_lower:
                context['has_list'] = True
                break
        
        # Check for question context
        for indicator in self.punctuation_contexts['question_context']:
            if indicator in text_lower:
                context['has_question'] = True
                break
        
        # Check for emphasis context
        for indicator in self.punctuation_contexts['emphasis_context']:
            if indicator in text_lower:
                context['has_emphasis'] = True
                break
        
        # Check if at sentence end
        context['sentence_end'] = any(word in text_lower 
                                    for word in self.punctuation_contexts['sentence_end'])
        
        return context
    
    def _calculate_confidence(self, matched_text: str, rule: ConversionRule, 
                            full_text: str, start: int, end: int, 
                            context_analysis: Dict) -> float:
        """Calculate confidence score for conversion"""
        confidence = 0.5  # Base confidence
        
        # Rule priority contributes to confidence
        confidence += rule.priority * 0.05
        
        # Context analysis contributions
        if context_analysis:
            # Mathematical context boosts math operator confidence
            if rule.replacement in '+-×÷=<>%' and context_analysis.get('mathematical_context', {}).get('has_numbers'):
                confidence += 0.2
            
            # Question context boosts question mark confidence
            if rule.replacement == '?' and context_analysis.get('punctuation_context', {}).get('has_question'):
                confidence += 0.3
            
            # List context boosts comma confidence
            if rule.replacement == ',' and context_analysis.get('punctuation_context', {}).get('has_list'):
                confidence += 0.2
        
        # Context requirements check
        if rule.context_required:
            context_words = full_text.lower().split()
            required_found = any(req in ' '.join(context_words) for req in rule.context_required)
            if required_found:
                confidence += 0.2
            else:
                confidence -= 0.1
        
        # Context forbidden check
        if rule.context_forbidden:
            context_words = full_text.lower().split()
            forbidden_found = any(forb in ' '.join(context_words) for forb in rule.context_forbidden)
            if forbidden_found:
                confidence -= 0.3
        
        # Position-based confidence (avoid sentence beginnings for some symbols)
        if start == 0 and rule.replacement in '+-×÷':
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _post_process_corrections(self, text: str) -> str:
        """Apply post-processing corrections"""
        # Fix common spacing issues
        text = re.sub(r'\s+([+\-×÷=<>%$@#&*()[\]{},.!?;:])', r' \1', text)
        text = re.sub(r'([+\-×÷=<>%$@#&*()[\]{},.!?;:])\s+', r'\1 ', text)
        
        # Fix multiple consecutive operators
        text = re.sub(r'([+\-×÷=<>])\s*\1+', r'\1', text)
        
        # Fix spacing around decimal points
        text = re.sub(r'(\d)\s+\.\s+(\d)', r'\1.\2', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def batch_convert(self, texts: List[str], confidence_threshold: float = 0.7) -> List[Tuple[str, Dict]]:
        """Convert multiple texts efficiently"""
        results = []
        for text in texts:
            converted, metadata = self.convert_text(text, confidence_threshold)
            results.append((converted, metadata))
        return results
    
    def get_conversion_statistics(self, texts: List[str]) -> Dict:
        """Get statistics about conversions for evaluation"""
        stats = {
            'total_texts': len(texts),
            'texts_with_conversions': 0,
            'total_conversions': 0,
            'conversion_types': defaultdict(int),
            'average_confidence': 0.0
        }
        
        total_confidence = 0.0
        conversion_count = 0
        
        for text in texts:
            _, metadata = self.convert_text(text)
            conversions = metadata.get('conversions', [])
            
            if conversions:
                stats['texts_with_conversions'] += 1
                stats['total_conversions'] += len(conversions)
                
                for conv in conversions:
                    stats['conversion_types'][conv['converted']] += 1
                    total_confidence += conv['confidence']
                    conversion_count += 1
        
        if conversion_count > 0:
            stats['average_confidence'] = total_confidence / conversion_count
        
        stats['conversion_rate'] = stats['texts_with_conversions'] / len(texts)
        
        return dict(stats) 