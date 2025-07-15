from reid_system import PlayerReIDSystem

def main():
    reid_system = PlayerReIDSystem(
        model_path='yolov8n.pt',
        conf_threshold=0.5,
        max_disappeared=30,
        stabilization_frames=3
    )

    video_path = "input_video.mp4"
    output_video = "output_reid.mp4"
    results_file = "reid_results.json"

    try:
        results = reid_system.process_video(
            video_path=video_path,
            output_path=output_video,
            show_display=True
        )

        reid_system.save_results(results, results_file)

        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Average FPS: {results['processing_stats']['avg_fps']:.1f}")
        print(f"Total unique IDs: {results['tracking_stats']['total_unique_ids']}")
        print(f"Max simultaneous tracks: {results['tracking_stats']['max_simultaneous_tracks']}")
        print(f"Results saved to: {results_file}")
        print(f"Output video: {output_video}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
