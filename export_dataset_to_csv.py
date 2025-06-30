"""
LibriHeavy Dataset को CSV format में export करने के लिए script
"""

import pandas as pd
import os
from datasets import load_dataset
from speech2symbol.data.dataset_loader import OperatorDatasetLoader
import logging
from typing import Dict, List
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_libriheavy_to_csv(
    output_file: str = "libriheavy_dataset.csv",
    subset_percentage: float = 0.1,  # Default 10% for faster processing
    split: str = "train",
    operator_focus: bool = True,
    include_audio_info: bool = True
):
    """
    LibriHeavy dataset को CSV format में export करता है
    
    Args:
        output_file: Output CSV file का नाम
        subset_percentage: Dataset का कितना हिस्सा export करना है (0.1 = 10%)
        split: कौन सा split use करना है ("train", "test", "validation")
        operator_focus: सिर्फ operator-heavy samples export करना है या नहीं
        include_audio_info: Audio file की information include करनी है या नहीं
    """
    
    logger.info(f"LibriHeavy dataset को CSV में export कर रहे हैं...")
    logger.info(f"Split: {split}, Subset: {subset_percentage*100}%")
    
    try:
        # Dataset loader initialize करें
        loader = OperatorDatasetLoader(operator_focus=operator_focus)
        
        # Dataset load करें
        dataset = loader.load_dataset(
            subset_percentage=subset_percentage,
            split=split
        )
        
        logger.info(f"Dataset loaded: {len(dataset)} samples")
        
        # अगर कोई samples नहीं मिले तो warning दें और सभी samples try करें
        if len(dataset) == 0:
            logger.warning("No operator-heavy samples found! Trying without operator filtering...")
            loader = OperatorDatasetLoader(operator_focus=False)
            dataset = loader.load_dataset(
                subset_percentage=subset_percentage,
                split=split
            )
            logger.info(f"Dataset loaded without filtering: {len(dataset)} samples")
        
        # अगर अभी भी कोई samples नहीं हैं तो error
        if len(dataset) == 0:
            logger.error("No samples found in dataset!")
            return None
        
        # CSV के लिए data prepare करें
        csv_data = []
        
        for i, sample in enumerate(dataset):
            try:
                # Basic information
                row_data = {
                    'sample_id': i,
                    'text': sample.get('text', ''),
                    'text_length': len(sample.get('text', '')),
                }
                
                # Audio information if requested
                if include_audio_info and 'audio' in sample:
                    audio_info = sample['audio']
                    row_data.update({
                        'audio_sampling_rate': audio_info.get('sampling_rate', ''),
                        'audio_array_length': len(audio_info.get('array', [])),
                        'audio_duration_seconds': len(audio_info.get('array', [])) / audio_info.get('sampling_rate', 1),
                    })
                
                # Other metadata
                for key, value in sample.items():
                    if key not in ['audio', 'text'] and not key.startswith('__'):
                        # Convert complex objects to string
                        if isinstance(value, (list, dict)):
                            row_data[f'meta_{key}'] = str(value)
                        else:
                            row_data[f'meta_{key}'] = value
                
                # Check for operators in text
                text_lower = sample.get('text', '').lower()
                operators_found = []
                for operator_word, symbol in loader.operator_mappings.items():
                    if operator_word in text_lower:
                        operators_found.append(f"{operator_word}→{symbol}")
                
                row_data['operators_found'] = '; '.join(operators_found) if operators_found else 'None'
                row_data['operator_count'] = len(operators_found)
                
                # Symbol count in text
                symbol_chars = "+-×÷=<>%$@#&*()[]{},.!?;:"
                row_data['symbol_count'] = sum(1 for char in sample.get('text', '') if char in symbol_chars)
                
                csv_data.append(row_data)
                
                # Progress logging
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(dataset)} samples...")
                    
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        # DataFrame बनाएं और CSV में save करें
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"✅ Dataset successfully exported to {output_file}")
        logger.info(f"📊 Total samples: {len(df)}")
        logger.info(f"📋 Columns: {list(df.columns)}")
        
        # Summary statistics - handle empty DataFrame
        print("\n📈 Dataset Summary:")
        print(f"Total samples: {len(df)}")
        
        if len(df) > 0:
            print(f"Average text length: {df['text_length'].mean():.1f} characters")
            
            if 'audio_duration_seconds' in df.columns:
                print(f"Average audio duration: {df['audio_duration_seconds'].mean():.2f} seconds")
                print(f"Total audio duration: {df['audio_duration_seconds'].sum()/3600:.2f} hours")
            
            print(f"Samples with operators: {df[df['operator_count'] > 0].shape[0]}")
            print(f"Samples with symbols: {df[df['symbol_count'] > 0].shape[0]}")
            
            # Top operators found
            print(f"\n🔤 Most common operators found:")
            all_operators = []
            for ops in df['operators_found']:
                if ops != 'None':
                    all_operators.extend(ops.split('; '))
            
            if all_operators:
                from collections import Counter
                top_operators = Counter(all_operators).most_common(10)
                for op, count in top_operators:
                    print(f"  {op}: {count} times")
            else:
                print("  No operators found in the dataset")
        else:
            print("No samples to analyze")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error exporting dataset: {e}")
        raise


def export_with_audio_paths(
    output_file: str = "libriheavy_with_paths.csv",
    subset_percentage: float = 0.05,
    split: str = "train"
):
    """
    Original dataset को audio file paths के साथ export करता है
    """
    logger.info("Original dataset with audio paths को export कर रहे हैं...")
    
    # Direct dataset load (without preprocessing)
    dataset = load_dataset("pkufool/libriheavy_long", split=split, streaming=False)
    
    if subset_percentage < 1.0:
        num_samples = int(len(dataset) * subset_percentage)
        dataset = dataset.select(range(num_samples))
    
    csv_data = []
    for i, sample in enumerate(dataset):
        row_data = {
            'sample_id': i,
            'text': sample.get('text', ''),
            'speaker_id': sample.get('speaker_id', ''),
            'chapter_id': sample.get('chapter_id', ''),
            'id': sample.get('id', ''),
        }
        
        # Audio info
        if 'audio' in sample:
            audio = sample['audio']
            row_data.update({
                'audio_path': audio.get('path', ''),
                'sampling_rate': audio.get('sampling_rate', ''),
                'audio_length': len(audio.get('array', [])),
                'duration_seconds': len(audio.get('array', [])) / audio.get('sampling_rate', 1) if audio.get('sampling_rate') else 0
            })
        
        csv_data.append(row_data)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(dataset)} samples...")
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    logger.info(f"✅ Dataset with audio paths exported to {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LibriHeavy dataset को CSV में export करें")
    parser.add_argument("--output", "-o", default="libriheavy_dataset.csv", help="Output CSV file name")
    parser.add_argument("--subset", "-s", type=float, default=0.1, help="Dataset का हिस्सा (0.1 = 10%)")
    parser.add_argument("--split", default="train", choices=["train", "test", "validation"], help="Dataset split")
    parser.add_argument("--no-operator-focus", action="store_true", help="सभी samples include करें, सिर्फ operator-heavy नहीं")
    parser.add_argument("--with-audio-paths", action="store_true", help="Audio file paths के साथ export करें")
    
    args = parser.parse_args()
    
    if args.with_audio_paths:
        export_with_audio_paths(
            output_file=args.output,
            subset_percentage=args.subset,
            split=args.split
        )
    else:
        export_libriheavy_to_csv(
            output_file=args.output,
            subset_percentage=args.subset,
            split=args.split,
            operator_focus=not args.no_operator_focus
        ) 