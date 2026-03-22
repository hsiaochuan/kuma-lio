#!/usr/bin/env python3
"""
ROS Bag Merger Script
This script merges multiple ROS bag files into a single output bag file.
"""

import rosbag
import os
import argparse
from tqdm import tqdm


def merge_rosbags(input_bag_paths: list, output_bag_fname: list, topics: list):
    """
    Merge multiple ROS bag files into a single output bag file.

    Args:
        input_bag_paths (list): List of paths to input bag files
        output_bag_fname (str): Path for the output merged bag file
    """

    print(f"Merging {len(input_bag_paths)} bag files into: {output_bag_fname}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_bag_fname)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Count total messages for progress tracking
    total_messages = 0
    print("Counting total messages in all bags...")
    for bag_path in input_bag_paths:
        if not os.path.exists(bag_path):
            print(f"Warning: File {bag_path} does not exist, skipping")
            continue
        try:
            with rosbag.Bag(bag_path, 'r') as bag:
                total_messages += bag.get_message_count()
        except Exception as e:
            print(f"Error reading {bag_path}: {e}")

    print(f"Total messages to process: {total_messages}")

    # Merge all bag files
    with rosbag.Bag(output_bag_fname, 'w') as output_bag:
        processed_messages = 0

        for bag_index, bag_path in enumerate(input_bag_paths, 1):
            if not os.path.exists(bag_path):
                continue

            print(f"\nProcessing bag {bag_index}/{len(input_bag_paths)}: {os.path.basename(bag_path)}")

            try:
                with rosbag.Bag(bag_path, 'r') as input_bag:
                    # Get bag info for progress reporting
                    bag_message_count = input_bag.get_message_count()

                    # Create progress bar for current bag
                    with tqdm(total=bag_message_count,
                              desc=f"Bag {bag_index}",
                              unit="msg") as pbar:
                        # Read all messages from current bag
                        for topic, msg, timestamp in input_bag.read_messages():
                            if topics is None or topic in topics:
                                # Write message to output bag with original timestamp
                                output_bag.write(topic, msg, timestamp)
                                processed_messages += 1
                                pbar.update(1)
                            else:
                                processed_messages += 1
                                pbar.update(1)

            except Exception as e:
                print(f"Error processing {bag_path}: {e}")
                continue

    # Verify the merged bag
    print(f"\nVerifying merged bag: {output_bag_fname}")
    try:
        with rosbag.Bag(output_bag_fname, 'r') as merged_bag:
            merged_count = merged_bag.get_message_count()
            merged_topics = merged_bag.get_type_and_topic_info()[1].keys()

            print(f"Merged bag created successfully!")
            print(f"Total messages in merged bag: {merged_count}")
            print(f"Topics in merged bag: {list(merged_topics)}")

    except Exception as e:
        print(f"Error verifying merged bag: {e}")


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description='Merge multiple ROS bag files into a single bag file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python merge_bags.py -o merged.bag -i bag1.bag bag2.bag bag3.bag
  python merge_bags.py --output merged_result.bag --input *.bag
        '''
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output path for the merged bag file'
    )

    parser.add_argument(
        '--topics',
        nargs='+',
        help='topics for merge'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        nargs='+',
        help='Input bag files to merge'
    )

    args = parser.parse_args()

    # Check if input files exist
    valid_bag_fnams = []
    for bag_fname in args.input:
        if os.path.exists(bag_fname):
            valid_bag_fnams.append(bag_fname)
        else:
            print(f"Warning: Input file {bag_fname} does not exist, skipping")

    if not valid_bag_fnams:
        print("Error: No valid input bag files provided")
        return

    # Merge the bags
    merge_rosbags(valid_bag_fnams, args.output, args.topics)


if __name__ == "__main__":
    main()
