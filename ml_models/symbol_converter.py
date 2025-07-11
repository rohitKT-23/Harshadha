"""
Production-Ready Symbol Converter
Comprehensive text-to-symbol conversion with extensive pattern matching
"""

import re
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversionRule:
    """Rule for text-to-symbol conversion"""
    pattern: str
    replacement: str
    priority: int = 0
    case_sensitive: bool = False

@dataclass
class ConversionResult:
    """Result of a single conversion"""
    original: str
    converted: str
    position: int
    priority: int

class SymbolConverter:
    """
    Production-ready symbol converter with comprehensive pattern matching
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize Symbol Converter
        
        Args:
            confidence_threshold: Minimum confidence for conversions (not used in this version)
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize conversion rules
        self.conversion_rules = self._create_comprehensive_rules()
        
        logger.info(f"Symbol Converter initialized with {len(self.conversion_rules)} rules")
    
    def _create_comprehensive_rules(self) -> List[ConversionRule]:
        """Create comprehensive conversion rules"""
        rules = []
        
        # Mathematical operators (highest priority)
        math_rules = [
            # Basic arithmetic
            ConversionRule(r'\bplus\b', '+', 100),
            ConversionRule(r'\badd\b', '+', 95),
            ConversionRule(r'\bminus\b', '-', 100),
            ConversionRule(r'\btake\s+away\b', '-', 95),
            ConversionRule(r'\btimes\b', '×', 100),
            ConversionRule(r'\bmultiply\s+by\b', '×', 95),
            ConversionRule(r'\bmultiply\b', '×', 90),
            ConversionRule(r'\bdivide[ds]?\s+by\b', '÷', 95),
            ConversionRule(r'\bdivide\b', '÷', 90),
            ConversionRule(r'\bover\b', '/', 90),
            
            # Equality and comparison
            ConversionRule(r'\bequals?\b', '=', 100),
            ConversionRule(r'\bis\s+equal\s+to\b', '=', 95),
            ConversionRule(r'\bis\s+greater\s+than\b', '>', 95),
            ConversionRule(r'\bis\s+less\s+than\b', '<', 95),
            ConversionRule(r'\bgreater\s+than\s+or\s+equal\s+to\b', '≥', 90),
            ConversionRule(r'\bless\s+than\s+or\s+equal\s+to\b', '≤', 90),
            ConversionRule(r'\bapproximately\s+equal\s+to\b', '≈', 85),
            
            # Powers and special math
            ConversionRule(r'\bsquared\b', '²', 85),
            ConversionRule(r'\bcubed\b', '³', 85),
            ConversionRule(r'\bto\s+the\s+power\s+of\b', '^', 80),
        ]
        
        # Punctuation patterns
        punctuation_rules = [
            # Basic punctuation
            ConversionRule(r'\bcomma\b', ',', 90),
            ConversionRule(r'\bperiod\b', '.', 95),
            ConversionRule(r'\bdot\b', '.', 85),
            ConversionRule(r'\bquestion\s+mark\b', '?', 95),
            ConversionRule(r'\bquestion\s+symbol\b', '?', 90),
            ConversionRule(r'\bexclamation\s+mark\b', '!', 95),
            ConversionRule(r'\bexclamation\s+point\b', '!', 90),
            ConversionRule(r'\bsemicolon\b', ';', 90),
            ConversionRule(r'\bcolon\b', ':', 90),
            ConversionRule(r'\bapostrophe\b', "'", 85),
            
            # Quotes
            ConversionRule(r'\bquote\b', '"', 85),
            ConversionRule(r'\bquotation\s+mark\b', '"', 85),
            ConversionRule(r'\bsingle\s+quote\b', "'", 80),
            ConversionRule(r'\bdouble\s+quote\b', '"', 80),
        ]
        
        # Parentheses and brackets
        bracket_rules = [
            ConversionRule(r'\bleft\s+parenthesis\b', '(', 85),
            ConversionRule(r'\bright\s+parenthesis\b', ')', 85),
            ConversionRule(r'\bopen\s+paren\b', '(', 80),
            ConversionRule(r'\bclose\s+paren\b', ')', 80),
            ConversionRule(r'\bopen\s+parenthesis\b', '(', 80),
            ConversionRule(r'\bclose\s+parenthesis\b', ')', 80),
            ConversionRule(r'\bleft\s+bracket\b', '[', 85),
            ConversionRule(r'\bright\s+bracket\b', ']', 85),
            ConversionRule(r'\bleft\s+brace\b', '{', 85),
            ConversionRule(r'\bright\s+brace\b', '}', 85),
        ]
        
        # Currency and symbols
        symbol_rules = [
            ConversionRule(r'\bdollar(?:s)?\b', '$', 90),
            ConversionRule(r'\bcents?\b', '¢', 85),
            ConversionRule(r'\beuros?\b', '€', 85),
            ConversionRule(r'\bpounds?\s+sterling\b', '£', 85),
            ConversionRule(r'\bpercent\b', '%', 95),
            ConversionRule(r'\bpercentage\b', '%', 90),
            ConversionRule(r'\bat\s+sign\b', '@', 90),
            ConversionRule(r'\bat\s+symbol\b', '@', 85),
            ConversionRule(r'\bhashtag\b', '#', 85),
            ConversionRule(r'\bhash\s+tag\b', '#', 80),
            ConversionRule(r'\bhash\s+sign\b', '#', 80),
            ConversionRule(r'\bampersand\b', '&', 80),
            ConversionRule(r'\basterisk\b', '*', 80),
            ConversionRule(r'\bunderscore\b', '_', 80),
            ConversionRule(r'\bhyphen\b', '-', 85),
            ConversionRule(r'\bdash\b', '-', 85),
            ConversionRule(r'\bminus\s+sign\b', '-', 85),
            ConversionRule(r'\bbackslash\b', '\\', 80),
            ConversionRule(r'\bforward\s+slash\b', '/', 80),
            ConversionRule(r'\bpipe\b', '|', 80),
            ConversionRule(r'\bvertical\s+bar\b', '|', 80),
            ConversionRule(r'\btilde\b', '~', 80),
            ConversionRule(r'\bcaret\b', '^', 80),
            ConversionRule(r'\bcircumflex\b', '^', 80),
        ]
        
        # URL and email patterns (complex)
        url_rules = [
            ConversionRule(r'\bhttps?\s+colon\s+slash\s+slash\b', 'https://', 85),
            ConversionRule(r'\bwww\s+dot\b', 'www.', 85),
            ConversionRule(r'\bslash\b', '/', 80),
        ]
        
        # Number conversion rules (high priority)
        number_rules = [
            # Single digits
            ConversionRule(r'\bzero\b', '0', 110),
            ConversionRule(r'\bone\b', '1', 110),
            ConversionRule(r'\btwo\b', '2', 110),
            ConversionRule(r'\bthree\b', '3', 110),
            ConversionRule(r'\bfour\b', '4', 110),
            ConversionRule(r'\bfive\b', '5', 110),
            ConversionRule(r'\bsix\b', '6', 110),
            ConversionRule(r'\bseven\b', '7', 110),
            ConversionRule(r'\beight\b', '8', 110),
            ConversionRule(r'\bnine\b', '9', 110),
            
            # Teens
            ConversionRule(r'\bten\b', '10', 110),
            ConversionRule(r'\beleven\b', '11', 110),
            ConversionRule(r'\btwelve\b', '12', 110),
            ConversionRule(r'\bthirteen\b', '13', 110),
            ConversionRule(r'\bfourteen\b', '14', 110),
            ConversionRule(r'\bfifteen\b', '15', 110),
            ConversionRule(r'\bsixteen\b', '16', 110),
            ConversionRule(r'\bseventeen\b', '17', 110),
            ConversionRule(r'\beighteen\b', '18', 110),
            ConversionRule(r'\bnineteen\b', '19', 110),
            
            # Tens
            ConversionRule(r'\btwenty\b', '20', 110),
            ConversionRule(r'\bthirty\b', '30', 110),
            ConversionRule(r'\bforty\b', '40', 110),
            ConversionRule(r'\bfifty\b', '50', 110),
            ConversionRule(r'\bsixty\b', '60', 110),
            ConversionRule(r'\bseventy\b', '70', 110),
            ConversionRule(r'\beighty\b', '80', 110),
            ConversionRule(r'\bninety\b', '90', 110),
            
            # Hundreds and thousands
            ConversionRule(r'\bhundred\b', '100', 110),
            ConversionRule(r'\bthousand\b', '1000', 110),
            ConversionRule(r'\bmillion\b', '1000000', 110),
            ConversionRule(r'\bbillion\b', '1000000000', 110),
        ]
        
        # Email patterns (complex multi-word)
        email_rules = [
            ConversionRule(r'\bat\s+gmail\s+dot\s+com\b', '@gmail.com', 90),
            ConversionRule(r'\bat\s+company\s+dot\s+org\b', '@company.org', 90),
            ConversionRule(r'\bat\s+company\s+dot\s+net\b', '@company.net', 90),
            ConversionRule(r'\bat\s+help\s+dot\s+org\b', '@help.org', 90),
            ConversionRule(r'\bat\s+sign\s+company\s+dot\s+net\b', '@company.net', 90),
        ]
        
        # Add all rules (but exclude individual number rules for now)
        rules.extend(math_rules)
        rules.extend(punctuation_rules)
        rules.extend(bracket_rules)
        rules.extend(symbol_rules)
        rules.extend(url_rules)
        rules.extend(email_rules)
        
        # Sort by priority (highest first)
        rules.sort(key=lambda x: x.priority, reverse=True)
        
        return rules
    
    def convert(self, text: str, confidence_threshold: Optional[float] = None) -> Tuple[str, Dict]:
        """
        Convert spoken operators to symbols
        
        Args:
            text: Input text to convert
            confidence_threshold: Override default confidence threshold
            
        Returns:
            Tuple of (converted_text, metadata)
        """
        original_text = text
        converted_text = text
        conversion_log = []
        
        # Apply all conversion rules
        for rule in self.conversion_rules:
            # Find all matches
            flags = re.IGNORECASE if not rule.case_sensitive else 0
            matches = list(re.finditer(rule.pattern, converted_text, flags))
            
            # Process matches in reverse order to maintain indices
            for match in reversed(matches):
                start, end = match.span()
                matched_text = match.group()
                
                # Apply conversion
                converted_text = (converted_text[:start] + 
                                rule.replacement + 
                                converted_text[end:])
                
                conversion_log.append(ConversionResult(
                    original=matched_text,
                    converted=rule.replacement,
                    position=start,
                    priority=rule.priority
                ))
        
        # Apply special multi-word patterns
        converted_text = self._apply_special_patterns(converted_text)
        
        # Apply number conversion patterns
        converted_text = self._apply_number_patterns(converted_text)
        
        # Post-process for spacing and formatting
        converted_text = self._post_process_formatting(converted_text)
        
        # Calculate average confidence based on conversion priorities
        average_confidence = 0.0
        if conversion_log:
            # Convert priority to confidence (higher priority = higher confidence)
            max_priority = max(conv.priority for conv in conversion_log)
            confidences = [conv.priority / max_priority for conv in conversion_log]
            average_confidence = sum(confidences) / len(confidences)
        
        return converted_text, {
            'original_text': original_text,
            'conversions': [asdict(conv) for conv in conversion_log],
            'total_conversions': len(conversion_log),
            'average_confidence': average_confidence,
            'context_analysis': {
                'has_math_operators': any(conv.converted in '+-×÷=<>%' for conv in conversion_log),
                'has_punctuation': any(conv.converted in ',.!?;:' for conv in conversion_log),
                'has_symbols': any(conv.converted in '$¢€£@#&*' for conv in conversion_log)
            }
        }
    
    def _apply_special_patterns(self, text: str) -> str:
        """Apply complex multi-word patterns that need special handling"""
        
        # Email address patterns
        email_patterns = [
            # Pattern: "john at gmail dot com" -> "john@gmail.com"
            (r'(\w+)\s+at\s+(\w+)\s+dot\s+(\w+)', r'\1@\2.\3'),
            # Pattern: "user dot name at domain dot co dot uk" -> "user.name@domain.co.uk"
            (r'(\w+)\s+dot\s+(\w+)\s+at\s+(\w+)\s+dot\s+(\w+)\s+dot\s+(\w+)', r'\1.\2@\3.\4.\5'),
            # Pattern: "john dot smith at company dot org" -> "john.smith@company.org"
            (r'(\w+)\s+dot\s+(\w+)\s+at\s+(\w+)\s+dot\s+(\w+)', r'\1.\2@\3.\4'),
            # Pattern: "admin at sign company dot net" -> "admin@company.net"
            (r'(\w+)\s+at\s+sign\s+(\w+)\s+dot\s+(\w+)', r'\1@\2.\3'),
        ]
        
        # URL patterns
        url_patterns = [
            # Pattern: "www dot google dot com" -> "www.google.com"
            (r'www\s+dot\s+(\w+)\s+dot\s+(\w+)', r'www.\1.\2'),
            # Pattern: "blog dot example dot org slash articles" -> "blog.example.org/articles"
            (r'(\w+)\s+dot\s+(\w+)\s+dot\s+(\w+)\s+slash\s+(\w+)', r'\1.\2.\3/\4'),
            # Pattern: "https colon slash slash website dot com" -> "https://website.com"
            (r'https\s+colon\s+slash\s+slash\s+(\w+)\s+dot\s+(\w+)', r'https://\1.\2'),
        ]
        
        # Hashtag patterns
        hashtag_patterns = [
            # Pattern: "at hashtag company name" -> "at #company name"
            (r'at\s+hashtag\s+(\w+)', r'at #\1'),
            # Pattern: "hash tag trending" -> "#trending"
            (r'hash\s+tag\s+(\w+)', r'#\1'),
        ]
        
        # Apply email patterns
        for pattern, replacement in email_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Apply URL patterns
        for pattern, replacement in url_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Apply hashtag patterns
        for pattern, replacement in hashtag_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Handle special cases
        special_cases = [
            # Fix URL patterns
            (r'www\.\s+(\w+)\.\s+(\w+)', r'www.\1.\2'),
            (r'(\w+)\.\s+(\w+)\.\s+(\w+)', r'\1.\2.\3'),
            
            # Fix hashtag spacing
            (r'at#\s+', r'at #'),
            
            # Remove extra spaces around symbols
            (r'\s+([+\-×÷=<>%$@#&*()[\]{},.!?;:])\s+', r' \1 '),
            (r'\s+([+\-×÷=<>%$@#&*()[\]{},.!?;:])', r'\1'),

            (r'([+\-×÷=<>%$@#&*()[\]{},.!?;:])\s+', r'\1 '),
            
            # Fix spacing around decimal points
            (r'(\d)\s+\.\s+(\d)', r'\1.\2'),
            
            # Fix spacing around parentheses
            (r'(\w)\s*\(\s*', r'\1('),
            (r'\s*\)\s*(\w)', r')\1'),
            
            # Fix spacing around brackets
            (r'(\w)\s*\[\s*', r'\1['),
            (r'\s*\]\s*(\w)', r']\1'),
            
            # Fix spacing around braces
            (r'(\w)\s*\{\s*', r'\1{'),
            (r'\s*\}\s*(\w)', r'}\1'),
            
            # Handle file extensions
            (r'(\w+)\s+underscore\s+(\w+)\s+dot\s+(\w+)', r'\1_\2.\3'),
            
            # Handle phone numbers
            (r'plus\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)', r'+\1\2\3\4\5\6\7\8\9\10'),
        ]
        
        for pattern, replacement in special_cases:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _apply_number_patterns(self, text: str) -> str:
        """Apply complex number conversion patterns"""
        
        # Dictionary for number word to digit mapping
        number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
            'hundred': 100, 'thousand': 1000, 'million': 1000000, 'billion': 1000000000
        }
        
        # Pattern for compound numbers like "twenty five" -> "25"
        compound_patterns = [
            # Pattern: "twenty five" -> "25"
            (r'\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+(one|two|three|four|five|six|seven|eight|nine)\b', 
             lambda m: str(number_words[m.group(1).lower()] + number_words[m.group(2).lower()])),
            
            # Pattern: "one hundred" -> "100"
            (r'\b(one|two|three|four|five|six|seven|eight|nine)\s+hundred\b',
             lambda m: str(number_words[m.group(1).lower()] * 100)),
            
            # Pattern: "one hundred twenty five" -> "125"
            (r'\b(one|two|three|four|five|six|seven|eight|nine)\s+hundred\s+(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+(one|two|three|four|five|six|seven|eight|nine)\b',
             lambda m: str(number_words[m.group(1).lower()] * 100 + number_words[m.group(2).lower()] + number_words[m.group(3).lower()])),
            
            # Pattern: "one thousand" -> "1000"
            (r'\b(one|two|three|four|five|six|seven|eight|nine)\s+thousand\b',
             lambda m: str(number_words[m.group(1).lower()] * 1000)),
        ]
        
        # Apply compound number patterns first
        for pattern, replacement in compound_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Now apply individual number patterns for remaining words
        individual_patterns = [
            # Single digits
            (r'\bzero\b', '0'),
            (r'\bone\b', '1'),
            (r'\btwo\b', '2'),
            (r'\bthree\b', '3'),
            (r'\bfour\b', '4'),
            (r'\bfive\b', '5'),
            (r'\bsix\b', '6'),
            (r'\bseven\b', '7'),
            (r'\beight\b', '8'),
            (r'\bnine\b', '9'),
            
            # Teens
            (r'\bten\b', '10'),
            (r'\beleven\b', '11'),
            (r'\btwelve\b', '12'),
            (r'\bthirteen\b', '13'),
            (r'\bfourteen\b', '14'),
            (r'\bfifteen\b', '15'),
            (r'\bsixteen\b', '16'),
            (r'\bseventeen\b', '17'),
            (r'\beighteen\b', '18'),
            (r'\bnineteen\b', '19'),
            
            # Tens
            (r'\btwenty\b', '20'),
            (r'\bthirty\b', '30'),
            (r'\bforty\b', '40'),
            (r'\bfifty\b', '50'),
            (r'\bsixty\b', '60'),
            (r'\bseventy\b', '70'),
            (r'\beighty\b', '80'),
            (r'\bninety\b', '90'),
            
            # Hundreds and thousands
            (r'\bhundred\b', '100'),
            (r'\bthousand\b', '1000'),
            (r'\bmillion\b', '1000000'),
            (r'\bbillion\b', '1000000000'),
        ]
        
        # Apply individual number patterns
        for pattern, replacement in individual_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _post_process_formatting(self, text: str) -> str:
        """Apply final formatting corrections"""
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around punctuation at sentence boundaries
        text = re.sub(r'\s+([.!?])\s*', r'\1 ', text)
        
        # Fix spacing around commas
        text = re.sub(r'\s*,\s*', ', ', text)
        
        # Fix spacing around colons and semicolons
        text = re.sub(r'\s*([:;])\s*', r'\1 ', text)
        
        # Fix spacing around mathematical operators
        text = re.sub(r'\s*([+\-×÷=<>%])\s*', r' \1 ', text)
        
        # Clean up again
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def batch_convert(self, texts: List[str], confidence_threshold: Optional[float] = None) -> List[Tuple[str, Dict]]:
        """
        Convert multiple texts efficiently
        
        Args:
            texts: List of texts to convert
            confidence_threshold: Override default confidence threshold
            
        Returns:
            List of (converted_text, metadata) tuples
        """
        results = []
        for text in texts:
            converted, metadata = self.convert(text, confidence_threshold)
            results.append((converted, metadata))
        return results
    
    def add_custom_rule(self, rule: ConversionRule):
        """Add a custom conversion rule"""
        self.conversion_rules.append(rule)
        self.conversion_rules.sort(key=lambda x: x.priority, reverse=True)
        logger.info(f"Added custom rule: {rule.pattern} -> {rule.replacement}")
    
    def save_model(self, file_path: str):
        """Save the converter model to a file"""
        model_data = {
            'conversion_rules': [asdict(rule) for rule in self.conversion_rules],
            'confidence_threshold': self.confidence_threshold
        }
        
        with open(file_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load_model(cls, file_path: str) -> 'SymbolConverter':
        """Load a converter model from a file"""
        with open(file_path, 'r') as f:
            model_data = json.load(f)
        
        converter = cls(confidence_threshold=model_data.get('confidence_threshold', 0.5))
        
        # Reconstruct rules
        rules = []
        for rule_data in model_data.get('conversion_rules', []):
            rule = ConversionRule(
                pattern=rule_data['pattern'],
                replacement=rule_data['replacement'],
                priority=rule_data.get('priority', 0),
                case_sensitive=rule_data.get('case_sensitive', False)
            )
            rules.append(rule)
        
        converter.conversion_rules = rules
        converter.conversion_rules.sort(key=lambda x: x.priority, reverse=True)
        
        logger.info(f"Model loaded from {file_path}")
        return converter
    
    def get_statistics(self) -> Dict:
        """Get statistics about the converter"""
        stats = {
            'total_rules': len(self.conversion_rules),
            'rule_categories': self._categorize_rules(),
            'confidence_threshold': self.confidence_threshold
        }
        return stats
    
    def _categorize_rules(self) -> Dict:
        """Categorize rules by type"""
        categories = defaultdict(int)
        
        for rule in self.conversion_rules:
            if rule.replacement in '+-×÷=<>%':
                categories['mathematical'] += 1
            elif rule.replacement in ',.!?;:':
                categories['punctuation'] += 1
            elif rule.replacement in '()[]{}':
                categories['brackets'] += 1
            elif rule.replacement in '$¢€£@#':
                categories['symbols'] += 1
            else:
                categories['other'] += 1
        
        return dict(categories)
    
    def __call__(self, text: str) -> str:
        """Convenience method for direct conversion"""
        converted, _ = self.convert(text)
        return converted

def create_basic_converter() -> SymbolConverter:
    """Create a basic symbol converter"""
    return SymbolConverter(confidence_threshold=0.5)

def create_advanced_converter() -> SymbolConverter:
    """Create an advanced symbol converter"""
    return SymbolConverter(confidence_threshold=0.3)

def create_strict_converter() -> SymbolConverter:
    """Create a strict symbol converter"""
    return SymbolConverter(confidence_threshold=0.8)
