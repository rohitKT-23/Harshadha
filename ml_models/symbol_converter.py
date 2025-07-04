"""
Production-Ready Symbol Converter
Intelligent text-to-symbol conversion with context awareness and confidence scoring
"""

import re
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
import os

# Optional NLP dependencies
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    nltk = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversionRule:
    """Rule for text-to-symbol conversion"""
    pattern: str
    replacement: str
    context_required: Optional[List[str]] = None
    context_forbidden: Optional[List[str]] = None
    priority: int = 0
    case_sensitive: bool = False
    confidence_base: float = 1.0

@dataclass
class ConversionResult:
    """Result of a single conversion"""
    original: str
    converted: str
    position: int
    confidence: float
    rule_priority: int
    context_match: bool = False

class SymbolConverter:
    """
    Production-ready symbol converter with:
    - Context-aware conversions
    - Confidence scoring
    - Rule-based and ML-based approaches
    - Batch processing
    - Model serialization
    """
    
    def __init__(
        self, 
        use_spacy: bool = True,
        confidence_threshold: float = 0.7,
        custom_rules: Optional[List[ConversionRule]] = None
    ):
        """
        Initialize Symbol Converter
        
        Args:
            use_spacy: Use spaCy for advanced NLP analysis
            confidence_threshold: Minimum confidence for conversions
            custom_rules: Additional custom conversion rules
        """
        self.confidence_threshold = confidence_threshold
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.nlp = None
        
        # Initialize NLP components
        self._setup_nlp()
        
        # Initialize conversion rules
        self.conversion_rules = self._create_conversion_rules()
        
        # Add custom rules if provided
        if custom_rules:
            self.conversion_rules.extend(custom_rules)
            self._sort_rules()
        
        # Context indicators
        self._setup_context_indicators()
        
        logger.info(f"Symbol Converter initialized (spaCy: {self.use_spacy})")
    
    def _setup_nlp(self):
        """Setup NLP components"""
        if self.use_spacy and spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy model not found. Falling back to rule-based approach.")
                self.use_spacy = False
                self.nlp = None
        
        if NLTK_AVAILABLE and nltk:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('wordnet', quiet=True)
            except:
                logger.warning("NLTK data not available")
    
    def _setup_context_indicators(self):
        """Setup context indicators for better conversion decisions"""
        self.math_contexts = {
            'equation_indicators': ['equals', 'equal', 'is', 'makes', 'gives', 'results'],
            'calculation_words': ['calculate', 'compute', 'solve', 'find', 'determine'],
            'number_words': ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'],
            'units': ['percent', 'percentage', 'degrees', 'dollars', 'cents'],
            'comparison': ['greater', 'less', 'more', 'fewer', 'bigger', 'smaller']
        }
        
        self.punctuation_contexts = {
            'sentence_end': ['end', 'finish', 'close', 'conclude', 'stop'],
            'list_context': ['first', 'second', 'third', 'next', 'then', 'finally', 'items'],
            'question_context': ['ask', 'question', 'inquire', 'wonder', 'what', 'how', 'why'],
            'emphasis_context': ['important', 'urgent', 'critical', 'wow', 'amazing', 'incredible']
        }
        
        self.communication_contexts = {
            'email_context': ['email', 'address', 'send', 'contact', 'write'],
            'web_context': ['website', 'url', 'link', 'domain', 'site'],
            'document_context': ['document', 'file', 'text', 'write', 'type']
        }
    
    def _create_conversion_rules(self) -> List[ConversionRule]:
        """Create comprehensive conversion rules"""
        rules = []
        
        # Mathematical operators (high priority)
        math_rules = [
            ConversionRule(r'\bplus\b', '+', ['number', 'digit'], priority=10, confidence_base=1.0),
            ConversionRule(r'\badd(?:ed)?\s+to\b', '+', ['number'], priority=9, confidence_base=0.9),
            ConversionRule(r'\bminus\b', '-', ['number', 'digit'], priority=10, confidence_base=1.0),
            ConversionRule(r'\bsubtract(?:ed)?\s+(?:from)?\b', '-', ['number'], priority=9, confidence_base=0.9),
            ConversionRule(r'\btimes\b', '×', ['number', 'digit'], priority=10, confidence_base=1.0),
            ConversionRule(r'\bmultipli(?:ed|es?)\s+by\b', '×', ['number'], priority=9, confidence_base=0.9),
            ConversionRule(r'\bdivide[ds]?\s+by\b', '÷', ['number'], priority=9, confidence_base=0.9),
            ConversionRule(r'\bover\b', '/', ['number', 'fraction'], priority=8, confidence_base=0.8),
            ConversionRule(r'\bequals?\b', '=', ['number', 'result'], priority=10, confidence_base=1.0),
            ConversionRule(r'\bis\s+equal\s+to\b', '=', ['number'], priority=9, confidence_base=0.9),
            ConversionRule(r'\bgreater\s+than\b', '>', ['number', 'comparison'], priority=9, confidence_base=0.85),
            ConversionRule(r'\bless\s+than\b', '<', ['number', 'comparison'], priority=9, confidence_base=0.85),
            ConversionRule(r'\bgreater\s+than\s+or\s+equal\s+to\b', '≥', ['number'], priority=8, confidence_base=0.8),
            ConversionRule(r'\bless\s+than\s+or\s+equal\s+to\b', '≤', ['number'], priority=8, confidence_base=0.8),
            ConversionRule(r'\bpercent\b', '%', ['number', 'rate'], priority=10, confidence_base=1.0),
            ConversionRule(r'\bpercentage\b', '%', ['number', 'rate'], priority=9, confidence_base=0.9),
            # Exponentiation and power rules
            ConversionRule(r'\bpower\b', '^', ['number', 'exponent'], priority=10, confidence_base=1.0),
            ConversionRule(r'\braised\s+to\s+(?:the\s+)?power\b', '^', ['number'], priority=10, confidence_base=1.0),
            ConversionRule(r'\bto\s+the\s+power\s+of\b', '^', ['number'], priority=10, confidence_base=1.0),
            ConversionRule(r'\bexponent\b', '^', ['number', 'math'], priority=9, confidence_base=0.9),
            ConversionRule(r'\bsquared\b', '²', ['number'], priority=9, confidence_base=0.95),
            ConversionRule(r'\bcubed\b', '³', ['number'], priority=9, confidence_base=0.95),
            ConversionRule(r'\bsquare\s+root\b', '√', ['number'], priority=9, confidence_base=0.9),
            ConversionRule(r'\bcube\s+root\b', '∛', ['number'], priority=8, confidence_base=0.85),
        ]
        
        # Punctuation rules (medium priority)
        punctuation_rules = [
            ConversionRule(r'\bcomma\b', ',', ['list', 'pause'], priority=7, confidence_base=1.0),
            ConversionRule(r'\bperiod\b', '.', ['end', 'sentence'], priority=8, confidence_base=0.9),
            ConversionRule(r'\bdot\b', '.', ['decimal', 'abbreviation'], priority=6, confidence_base=0.8),
            ConversionRule(r'\bquestion\s+mark\b', '?', ['question', 'ask'], priority=8, confidence_base=0.95),
            ConversionRule(r'\bexclamation\s+(?:mark|point)\b', '!', ['emphasis', 'surprise'], priority=8, confidence_base=0.95),
            ConversionRule(r'\bsemicolon\b', ';', ['list', 'clause'], priority=7, confidence_base=0.9),
            ConversionRule(r'\bcolon\b', ':', ['list', 'explanation'], priority=7, confidence_base=0.9),
            ConversionRule(r'\bapostrophe\b', "'", ['possession', 'contraction'], priority=6, confidence_base=0.8),
            ConversionRule(r'\bquote\b', '"', ['quotation', 'speech'], priority=6, confidence_base=0.8),
            ConversionRule(r'\bquotation\s+mark\b', '"', ['quotation'], priority=7, confidence_base=0.9),
        ]
        
        # Parentheses and brackets
        bracket_rules = [
            ConversionRule(r'\bleft\s+parenthesis\b', '(', ['grouping'], priority=5, confidence_base=0.9),
            ConversionRule(r'\bright\s+parenthesis\b', ')', ['grouping'], priority=5, confidence_base=0.9),
            ConversionRule(r'\bopen\s+(?:paren|parenthesis)\b', '(', ['grouping'], priority=4, confidence_base=0.8),
            ConversionRule(r'\bclose\s+(?:paren|parenthesis)\b', ')', ['grouping'], priority=4, confidence_base=0.8),
            ConversionRule(r'\bleft\s+bracket\b', '[', ['array', 'index'], priority=5, confidence_base=0.8),
            ConversionRule(r'\bright\s+bracket\b', ']', ['array', 'index'], priority=5, confidence_base=0.8),
            ConversionRule(r'\bleft\s+brace\b', '{', ['set', 'code'], priority=4, confidence_base=0.7),
            ConversionRule(r'\bright\s+brace\b', '}', ['set', 'code'], priority=4, confidence_base=0.7),
        ]
        
        # Currency and symbols
        symbol_rules = [
            ConversionRule(r'\bdollar(?:s)?\b', '$', ['money', 'cost'], priority=9, confidence_base=1.0),
            ConversionRule(r'\bcents?\b', '¢', ['money', 'small'], priority=8, confidence_base=0.9),
            ConversionRule(r'\bat\s+sign\b', '@', ['email', 'location'], priority=8, confidence_base=0.95),
            ConversionRule(r'\bhashtag\b', '#', ['social', 'number'], priority=7, confidence_base=0.9),
            ConversionRule(r'\bampersand\b', '&', ['and', 'company'], priority=6, confidence_base=0.8),
            ConversionRule(r'\basterisk\b', '*', ['multiply', 'footnote'], priority=6, confidence_base=0.8),
            ConversionRule(r'\bunderscore\b', '_', ['space', 'code'], priority=5, confidence_base=0.7),
        ]
        
        # Comprehensive Keyboard Dictionary - All common keyboard symbols
        keyboard_rules = [
            # Basic punctuation
            ConversionRule(r'\bcomma\b', ',', ['list', 'pause'], priority=7, confidence_base=1.0),
            ConversionRule(r'\bperiod\b', '.', ['end', 'sentence'], priority=8, confidence_base=0.9),
            ConversionRule(r'\bdot\b', '.', ['decimal', 'abbreviation'], priority=6, confidence_base=0.8),
            ConversionRule(r'\bquestion\s+mark\b', '?', ['question', 'ask'], priority=8, confidence_base=0.95),
            ConversionRule(r'\bexclamation\s+(?:mark|point)\b', '!', ['emphasis', 'surprise'], priority=8, confidence_base=0.95),
            ConversionRule(r'\bsemicolon\b', ';', ['list', 'clause'], priority=7, confidence_base=0.9),
            ConversionRule(r'\bcolon\b', ':', ['list', 'explanation'], priority=7, confidence_base=0.9),
            ConversionRule(r'\bapostrophe\b', "'", ['possession', 'contraction'], priority=6, confidence_base=0.8),
            ConversionRule(r'\bquote\b', '"', ['quotation', 'speech'], priority=6, confidence_base=0.8),
            ConversionRule(r'\bquotation\s+mark\b', '"', ['quotation'], priority=7, confidence_base=0.9),
            ConversionRule(r'\bsingle\s+quote\b', "'", ['quotation'], priority=6, confidence_base=0.8),
            ConversionRule(r'\bdouble\s+quote\b', '"', ['quotation'], priority=7, confidence_base=0.9),
            
            # Parentheses and brackets
            ConversionRule(r'\bleft\s+parenthesis\b', '(', ['grouping'], priority=5, confidence_base=0.9),
            ConversionRule(r'\bright\s+parenthesis\b', ')', ['grouping'], priority=5, confidence_base=0.9),
            ConversionRule(r'\bopen\s+(?:paren|parenthesis)\b', '(', ['grouping'], priority=4, confidence_base=0.8),
            ConversionRule(r'\bclose\s+(?:paren|parenthesis)\b', ')', ['grouping'], priority=4, confidence_base=0.8),
            ConversionRule(r'\bleft\s+bracket\b', '[', ['array', 'index'], priority=5, confidence_base=0.8),
            ConversionRule(r'\bright\s+bracket\b', ']', ['array', 'index'], priority=5, confidence_base=0.8),
            ConversionRule(r'\bopen\s+bracket\b', '[', ['array'], priority=4, confidence_base=0.8),
            ConversionRule(r'\bclose\s+bracket\b', ']', ['array'], priority=4, confidence_base=0.8),
            ConversionRule(r'\bleft\s+brace\b', '{', ['set', 'code'], priority=4, confidence_base=0.7),
            ConversionRule(r'\bright\s+brace\b', '}', ['set', 'code'], priority=4, confidence_base=0.7),
            ConversionRule(r'\bopen\s+brace\b', '{', ['set'], priority=4, confidence_base=0.7),
            ConversionRule(r'\bclose\s+brace\b', '}', ['set'], priority=4, confidence_base=0.7),
            ConversionRule(r'\bcurly\s+brace\b', '{', ['set'], priority=4, confidence_base=0.7),
            
            # Mathematical operators
            ConversionRule(r'\bplus\b', '+', ['number', 'digit'], priority=10, confidence_base=1.0),
            ConversionRule(r'\badd(?:ed)?\s+to\b', '+', ['number'], priority=9, confidence_base=0.9),
            ConversionRule(r'\bminus\b', '-', ['number', 'digit'], priority=10, confidence_base=1.0),
            ConversionRule(r'\bsubtract(?:ed)?\s+(?:from)?\b', '-', ['number'], priority=9, confidence_base=0.9),
            ConversionRule(r'\btimes\b', '×', ['number', 'digit'], priority=10, confidence_base=1.0),
            ConversionRule(r'\bmultipli(?:ed|es?)\s+by\b', '×', ['number'], priority=9, confidence_base=0.9),
            ConversionRule(r'\bdivide[ds]?\s+by\b', '÷', ['number'], priority=9, confidence_base=0.9),
            ConversionRule(r'\bover\b', '/', ['number', 'fraction'], priority=8, confidence_base=0.8),
            ConversionRule(r'\bslash\b', '/', ['path', 'division'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bforward\s+slash\b', '/', ['path'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bbackslash\b', '\\', ['path', 'escape'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bequals?\b', '=', ['number', 'result'], priority=10, confidence_base=1.0),
            ConversionRule(r'\bis\s+equal\s+to\b', '=', ['number'], priority=9, confidence_base=0.9),
            ConversionRule(r'\bgreater\s+than\b', '>', ['number', 'comparison'], priority=9, confidence_base=0.85),
            ConversionRule(r'\bless\s+than\b', '<', ['number', 'comparison'], priority=9, confidence_base=0.85),
            ConversionRule(r'\bgreater\s+than\s+or\s+equal\s+to\b', '≥', ['number'], priority=8, confidence_base=0.8),
            ConversionRule(r'\bless\s+than\s+or\s+equal\s+to\b', '≤', ['number'], priority=8, confidence_base=0.8),
            ConversionRule(r'\bnot\s+equal\s+to\b', '≠', ['number'], priority=8, confidence_base=0.8),
            ConversionRule(r'\bapproximately\s+equal\b', '≈', ['number'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bpercent\b', '%', ['number', 'rate'], priority=10, confidence_base=1.0),
            ConversionRule(r'\bpercentage\b', '%', ['number', 'rate'], priority=9, confidence_base=0.9),
            
            # Exponentiation and power
            ConversionRule(r'\bpower\b', '^', ['number', 'exponent'], priority=10, confidence_base=1.0),
            ConversionRule(r'\braised\s+to\s+(?:the\s+)?power\b', '^', ['number'], priority=10, confidence_base=1.0),
            ConversionRule(r'\bto\s+the\s+power\s+of\b', '^', ['number'], priority=10, confidence_base=1.0),
            ConversionRule(r'\bexponent\b', '^', ['number', 'math'], priority=9, confidence_base=0.9),
            ConversionRule(r'\bsquared\b', '²', ['number'], priority=9, confidence_base=0.95),
            ConversionRule(r'\bcubed\b', '³', ['number'], priority=9, confidence_base=0.95),
            ConversionRule(r'\bsquare\s+root\b', '√', ['number'], priority=9, confidence_base=0.9),
            ConversionRule(r'\bcube\s+root\b', '∛', ['number'], priority=8, confidence_base=0.85),
            ConversionRule(r'\bcaret\b', '^', ['exponent'], priority=8, confidence_base=0.9),
            
            # Special symbols and characters
            ConversionRule(r'\bat\s+sign\b', '@', ['email', 'location'], priority=8, confidence_base=0.95),
            ConversionRule(r'\bat\s+symbol\b', '@', ['email'], priority=8, confidence_base=0.95),
            ConversionRule(r'\bhashtag\b', '#', ['social', 'number'], priority=7, confidence_base=0.9),
            ConversionRule(r'\bnumber\s+sign\b', '#', ['number'], priority=7, confidence_base=0.9),
            ConversionRule(r'\bpound\s+sign\b', '#', ['number'], priority=7, confidence_base=0.9),
            ConversionRule(r'\bampersand\b', '&', ['and', 'company'], priority=6, confidence_base=0.8),
            ConversionRule(r'\band\s+symbol\b', '&', ['and'], priority=6, confidence_base=0.8),
            ConversionRule(r'\basterisk\b', '*', ['multiply', 'footnote'], priority=6, confidence_base=0.8),
            ConversionRule(r'\bstar\b', '*', ['multiply'], priority=6, confidence_base=0.8),
            ConversionRule(r'\bunderscore\b', '_', ['space', 'code'], priority=5, confidence_base=0.7),
            ConversionRule(r'\bunder\s+score\b', '_', ['space'], priority=5, confidence_base=0.7),
            ConversionRule(r'\bpipe\b', '|', ['or', 'separator'], priority=6, confidence_base=0.8),
            ConversionRule(r'\bvertical\s+bar\b', '|', ['separator'], priority=6, confidence_base=0.8),
            ConversionRule(r'\btilde\b', '~', ['approximate', 'home'], priority=5, confidence_base=0.7),
            ConversionRule(r'\bbacktick\b', '`', ['code', 'grave'], priority=5, confidence_base=0.7),
            ConversionRule(r'\bgrave\s+accent\b', '`', ['code'], priority=5, confidence_base=0.7),
            
            # Currency symbols
            ConversionRule(r'\bdollar(?:s)?\b', '$', ['money', 'cost'], priority=9, confidence_base=1.0),
            ConversionRule(r'\bdollar\s+sign\b', '$', ['money'], priority=9, confidence_base=1.0),
            ConversionRule(r'\bcents?\b', '¢', ['money', 'small'], priority=8, confidence_base=0.9),
            ConversionRule(r'\beuro\b', '€', ['money'], priority=8, confidence_base=0.9),
            ConversionRule(r'\bpound\b', '£', ['money'], priority=8, confidence_base=0.9),
            ConversionRule(r'\byen\b', '¥', ['money'], priority=7, confidence_base=0.8),
            ConversionRule(r'\brupee\b', '₹', ['money'], priority=7, confidence_base=0.8),
            
            # Programming and technical symbols
            ConversionRule(r'\bmodulo\b', '%', ['programming'], priority=8, confidence_base=0.9),
            ConversionRule(r'\bmod\b', '%', ['programming'], priority=8, confidence_base=0.9),
            ConversionRule(r'\bhash\b', '#', ['programming'], priority=7, confidence_base=0.9),
            ConversionRule(r'\bsharp\b', '#', ['music'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bflat\b', '♭', ['music'], priority=6, confidence_base=0.7),
            ConversionRule(r'\bnatural\b', '♮', ['music'], priority=6, confidence_base=0.7),
            
            # Arrow symbols
            ConversionRule(r'\bright\s+arrow\b', '→', ['direction'], priority=6, confidence_base=0.8),
            ConversionRule(r'\bleft\s+arrow\b', '←', ['direction'], priority=6, confidence_base=0.8),
            ConversionRule(r'\bup\s+arrow\b', '↑', ['direction'], priority=6, confidence_base=0.8),
            ConversionRule(r'\bdown\s+arrow\b', '↓', ['direction'], priority=6, confidence_base=0.8),
            ConversionRule(r'\bdouble\s+arrow\b', '⇒', ['implication'], priority=6, confidence_base=0.8),
            ConversionRule(r'\bleft\s+right\s+arrow\b', '↔', ['bidirectional'], priority=6, confidence_base=0.8),
            
            # Greek letters (common ones)
            ConversionRule(r'\balpha\b', 'α', ['greek'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bbeta\b', 'β', ['greek'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bgamma\b', 'γ', ['greek'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bdelta\b', 'δ', ['greek'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bepsilon\b', 'ε', ['greek'], priority=7, confidence_base=0.8),
            ConversionRule(r'\btheta\b', 'θ', ['greek'], priority=7, confidence_base=0.8),
            ConversionRule(r'\blambda\b', 'λ', ['greek'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bmu\b', 'μ', ['greek'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bpi\b', 'π', ['greek'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bsigma\b', 'σ', ['greek'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bphi\b', 'φ', ['greek'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bomega\b', 'ω', ['greek'], priority=7, confidence_base=0.8),
            
            # Set theory symbols
            ConversionRule(r'\bunion\b', '∪', ['set'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bintersection\b', '∩', ['set'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bsubset\b', '⊂', ['set'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bsuperset\b', '⊃', ['set'], priority=7, confidence_base=0.8),
            ConversionRule(r'\belement\s+of\b', '∈', ['set'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bnot\s+element\s+of\b', '∉', ['set'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bempty\s+set\b', '∅', ['set'], priority=7, confidence_base=0.8),
            
            # Logic symbols
            ConversionRule(r'\band\s+symbol\b', '∧', ['logic'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bor\s+symbol\b', '∨', ['logic'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bnot\s+symbol\b', '¬', ['logic'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bimplies\b', '⇒', ['logic'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bif\s+and\s+only\s+if\b', '⇔', ['logic'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bfor\s+all\b', '∀', ['logic'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bthere\s+exists\b', '∃', ['logic'], priority=7, confidence_base=0.8),
            
            # Calculus symbols
            ConversionRule(r'\bintegral\b', '∫', ['calculus'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bsummation\b', '∑', ['calculus'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bproduct\s+symbol\b', '∏', ['calculus'], priority=7, confidence_base=0.8),
            ConversionRule(r'\binfinity\b', '∞', ['calculus'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bpartial\b', '∂', ['calculus'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bgradient\b', '∇', ['calculus'], priority=7, confidence_base=0.8),
            
            # Other mathematical symbols
            ConversionRule(r'\btherefore\b', '∴', ['math'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bbecause\b', '∵', ['math'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bproportional\b', '∝', ['math'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bperpendicular\b', '⊥', ['math'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bparallel\b', '∥', ['math'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bangle\b', '∠', ['math'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bdegree\b', '°', ['math'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bprime\b', '′', ['math'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bdouble\s+prime\b', '″', ['math'], priority=7, confidence_base=0.8),
            
            # Typography symbols
            ConversionRule(r'\bcopyright\b', '©', ['legal'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bregistered\b', '®', ['legal'], priority=7, confidence_base=0.8),
            ConversionRule(r'\btrademark\b', '™', ['legal'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bsection\b', '§', ['legal'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bparagraph\b', '¶', ['legal'], priority=7, confidence_base=0.8),
            ConversionRule(r'\bbullet\b', '•', ['list'], priority=6, confidence_base=0.7),
            ConversionRule(r'\bdagger\b', '†', ['footnote'], priority=6, confidence_base=0.7),
            ConversionRule(r'\bdouble\s+dagger\b', '‡', ['footnote'], priority=6, confidence_base=0.7),
            
            # Card symbols
            ConversionRule(r'\bspade\b', '♠', ['card'], priority=6, confidence_base=0.7),
            ConversionRule(r'\bheart\b', '♥', ['card'], priority=6, confidence_base=0.7),
            ConversionRule(r'\bdiamond\b', '♦', ['card'], priority=6, confidence_base=0.7),
            ConversionRule(r'\bclub\b', '♣', ['card'], priority=6, confidence_base=0.7),
            
            # Chess symbols
            ConversionRule(r'\bwhite\s+king\b', '♔', ['chess'], priority=6, confidence_base=0.7),
            ConversionRule(r'\bwhite\s+queen\b', '♕', ['chess'], priority=6, confidence_base=0.7),
            ConversionRule(r'\bwhite\s+rook\b', '♖', ['chess'], priority=6, confidence_base=0.7),
            ConversionRule(r'\bwhite\s+bishop\b', '♗', ['chess'], priority=6, confidence_base=0.7),
            ConversionRule(r'\bwhite\s+knight\b', '♘', ['chess'], priority=6, confidence_base=0.7),
            ConversionRule(r'\bwhite\s+pawn\b', '♙', ['chess'], priority=6, confidence_base=0.7),
            ConversionRule(r'\bblack\s+king\b', '♚', ['chess'], priority=6, confidence_base=0.7),
            ConversionRule(r'\bblack\s+queen\b', '♛', ['chess'], priority=6, confidence_base=0.7),
            ConversionRule(r'\bblack\s+rook\b', '♜', ['chess'], priority=6, confidence_base=0.7),
            ConversionRule(r'\bblack\s+bishop\b', '♝', ['chess'], priority=6, confidence_base=0.7),
            ConversionRule(r'\bblack\s+knight\b', '♞', ['chess'], priority=6, confidence_base=0.7),
            ConversionRule(r'\bblack\s+pawn\b', '♟', ['chess'], priority=6, confidence_base=0.7),
        ]
        
        rules.extend(math_rules)
        rules.extend(punctuation_rules)
        rules.extend(bracket_rules)
        rules.extend(symbol_rules)
        rules.extend(keyboard_rules)
        
        # Sort rules by priority (highest first)
        rules.sort(key=lambda x: x.priority, reverse=True)
        return rules
    
    def _sort_rules(self):
        """Sort rules by priority"""
        if hasattr(self, 'conversion_rules'):
            self.conversion_rules.sort(key=lambda x: x.priority, reverse=True)
    
    def convert(
        self, 
        text: str, 
        confidence_threshold: Optional[float] = None
    ) -> Tuple[str, Dict]:
        """
        Convert spoken operators to symbols
        
        Args:
            text: Input text to convert
            confidence_threshold: Minimum confidence for conversion (optional)
            
        Returns:
            Tuple of (converted_text, metadata)
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        original_text = text
        converted_text = text
        conversions = []
        
        # Analyze context if spaCy is available
        context_analysis = self._analyze_context(text) if self.use_spacy else {}
        
        # Apply conversion rules
        for rule in self.conversion_rules:
            matches = list(re.finditer(rule.pattern, converted_text, 
                                     re.IGNORECASE if not rule.case_sensitive else 0))
            
            for match in reversed(matches):  # Reverse to maintain indices
                start, end = match.span()
                matched_text = match.group()
                
                # Calculate confidence
                confidence = self._calculate_confidence(
                    matched_text, rule, converted_text, start, end, context_analysis
                )
                
                if confidence >= confidence_threshold:
                    # Apply conversion
                    converted_text = (converted_text[:start] + 
                                    rule.replacement + 
                                    converted_text[end:])
                    
                    conversions.append(ConversionResult(
                        original=matched_text,
                        converted=rule.replacement,
                        position=start,
                        confidence=confidence,
                        rule_priority=rule.priority,
                        context_match=True
                    ))
        
        # Post-process for common issues
        converted_text = self._post_process(converted_text)
        
        # Calculate statistics
        total_conversions = len(conversions)
        avg_confidence = sum(c.confidence for c in conversions) / max(total_conversions, 1)
        
        return converted_text, {
            'original_text': original_text,
            'conversions': [asdict(c) for c in conversions],
            'context_analysis': context_analysis,
            'total_conversions': total_conversions,
            'average_confidence': avg_confidence,
            'confidence_threshold': confidence_threshold
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
            'word_count': len([token for token in doc if not token.is_space]),
            'math_context': self._detect_mathematical_context(doc),
            'punctuation_context': self._detect_punctuation_context(doc),
            'communication_context': self._detect_communication_context(doc)
        }
        
        return analysis
    
    def _detect_mathematical_context(self, doc) -> Dict:
        """Detect mathematical context in text"""
        tokens = [token.text.lower() for token in doc]
        
        context = {
            'has_equation_indicators': any(word in tokens for word in self.math_contexts['equation_indicators']),
            'has_calculation_words': any(word in tokens for word in self.math_contexts['calculation_words']),
            'has_number_words': any(word in tokens for word in self.math_contexts['number_words']),
            'has_units': any(word in tokens for word in self.math_contexts['units']),
            'has_comparison': any(word in tokens for word in self.math_contexts['comparison']),
            'has_numeric_entities': any(ent.label_ in ['CARDINAL', 'QUANTITY'] for ent in doc.ents)
        }
        
        context['is_mathematical'] = any(context.values())
        return context
    
    def _detect_punctuation_context(self, doc) -> Dict:
        """Detect punctuation context in text"""
        tokens = [token.text.lower() for token in doc]
        
        context = {
            'has_sentence_end': any(word in tokens for word in self.punctuation_contexts['sentence_end']),
            'has_list_context': any(word in tokens for word in self.punctuation_contexts['list_context']),
            'has_question_context': any(word in tokens for word in self.punctuation_contexts['question_context']),
            'has_emphasis_context': any(word in tokens for word in self.punctuation_contexts['emphasis_context'])
        }
        
        return context
    
    def _detect_communication_context(self, doc) -> Dict:
        """Detect communication context in text"""
        tokens = [token.text.lower() for token in doc]
        
        context = {
            'has_email_context': any(word in tokens for word in self.communication_contexts['email_context']),
            'has_web_context': any(word in tokens for word in self.communication_contexts['web_context']),
            'has_document_context': any(word in tokens for word in self.communication_contexts['document_context'])
        }
        
        return context
    
    def _calculate_confidence(
        self, 
        matched_text: str, 
        rule: ConversionRule, 
        full_text: str, 
        start: int, 
        end: int, 
        context_analysis: Dict
    ) -> float:
        """Calculate confidence score for conversion"""
        base_confidence = rule.confidence_base
        
        # Context matching bonus
        context_bonus = 0.0
        if rule.context_required and context_analysis:
            # Check if required context is present
            if self._check_context_requirements(rule.context_required, context_analysis):
                context_bonus = 0.2
        
        # Forbidden context penalty
        context_penalty = 0.0
        if rule.context_forbidden and context_analysis:
            if self._check_context_forbidden(rule.context_forbidden, context_analysis):
                context_penalty = 0.3
        
        # Position-based adjustments
        position_bonus = 0.0
        if self._is_good_position(matched_text, full_text, start, end):
            position_bonus = 0.1
        
        # Calculate final confidence
        confidence = base_confidence + context_bonus + position_bonus - context_penalty
        return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
    
    def _check_context_requirements(self, requirements: List[str], context: Dict) -> bool:
        """Check if context requirements are met"""
        for req in requirements:
            if req == 'number' and context.get('has_numbers'):
                return True
            elif req == 'math' and context.get('math_context', {}).get('is_mathematical'):
                return True
            elif req == 'list' and context.get('punctuation_context', {}).get('has_list_context'):
                return True
            elif req == 'question' and context.get('punctuation_context', {}).get('has_question_context'):
                return True
            elif req == 'email' and context.get('communication_context', {}).get('has_email_context'):
                return True
        return False
    
    def _check_context_forbidden(self, forbidden: List[str], context: Dict) -> bool:
        """Check if forbidden context is present"""
        # Similar logic to requirements but for forbidden contexts
        return False  # Simplified for now
    
    def _is_good_position(self, matched_text: str, full_text: str, start: int, end: int) -> bool:
        """Check if position is good for conversion"""
        # Simple heuristics for position-based confidence
        before_text = full_text[:start].strip()
        after_text = full_text[end:].strip()
        
        # Check for number patterns around operators
        if matched_text.lower() in ['plus', 'minus', 'times', 'equals']:
            # Look for numbers before and after
            if (before_text and before_text[-1].isdigit()) or \
               (after_text and after_text[0].isdigit()):
                return True
        
        return False
    
    def _post_process(self, text: str) -> str:
        """Post-process the converted text"""
        # Fix common spacing issues
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.;:!?])([^\s])', r'\1 \2', text)  # Add space after punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        return text
    
    def batch_convert(
        self, 
        texts: List[str], 
        confidence_threshold: Optional[float] = None
    ) -> List[Tuple[str, Dict]]:
        """Convert multiple texts"""
        results = []
        for text in texts:
            result = self.convert(text, confidence_threshold)
            results.append(result)
        return results
    
    def add_custom_rule(self, rule: ConversionRule):
        """Add a custom conversion rule"""
        self.conversion_rules.append(rule)
        self._sort_rules()
    
    def save_model(self, file_path: str):
        """Save converter configuration to file"""
        config = {
            'rules': [asdict(rule) for rule in self.conversion_rules],
            'confidence_threshold': self.confidence_threshold,
            'math_contexts': self.math_contexts,
            'punctuation_contexts': self.punctuation_contexts,
            'communication_contexts': self.communication_contexts
        }
        
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Symbol converter saved to: {file_path}")
    
    @classmethod
    def load_model(cls, file_path: str) -> 'SymbolConverter':
        """Load converter configuration from file"""
        with open(file_path, 'r') as f:
            config = json.load(f)
        
        # Reconstruct rules
        rules = [ConversionRule(**rule_data) for rule_data in config['rules']]
        
        # Create converter
        converter = cls(custom_rules=rules, confidence_threshold=config['confidence_threshold'])
        
        # Restore contexts
        converter.math_contexts = config['math_contexts']
        converter.punctuation_contexts = config['punctuation_contexts']
        converter.communication_contexts = config['communication_contexts']
        
        logger.info(f"Symbol converter loaded from: {file_path}")
        return converter
    
    def get_statistics(self) -> Dict:
        """Get converter statistics"""
        return {
            'total_rules': len(self.conversion_rules),
            'rule_categories': self._categorize_rules(),
            'confidence_threshold': self.confidence_threshold,
            'spacy_enabled': self.use_spacy,
            'context_indicators': {
                'math_contexts': len(self.math_contexts),
                'punctuation_contexts': len(self.punctuation_contexts),
                'communication_contexts': len(self.communication_contexts)
            }
        }
    
    def _categorize_rules(self) -> Dict:
        """Categorize rules by type"""
        categories = defaultdict(int)
        for rule in self.conversion_rules:
            if rule.replacement in ['+', '-', '×', '÷', '=', '>', '<', '≥', '≤', '%']:
                categories['mathematical'] += 1
            elif rule.replacement in [',', '.', '?', '!', ';', ':', "'", '"']:
                categories['punctuation'] += 1
            elif rule.replacement in ['(', ')', '[', ']', '{', '}']:
                categories['brackets'] += 1
            else:
                categories['symbols'] += 1
        return dict(categories)
    
    def __call__(self, text: str) -> str:
        """Make the class callable"""
        converted_text, _ = self.convert(text)
        return converted_text

# Factory functions for easy instantiation
def create_basic_converter() -> SymbolConverter:
    """Create basic symbol converter"""
    return SymbolConverter(use_spacy=False, confidence_threshold=0.7)

def create_advanced_converter() -> SymbolConverter:
    """Create advanced symbol converter with NLP"""
    return SymbolConverter(use_spacy=True, confidence_threshold=0.6)

def create_strict_converter() -> SymbolConverter:
    """Create strict converter with high confidence threshold"""
    return SymbolConverter(use_spacy=True, confidence_threshold=0.9)
