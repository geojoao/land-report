import subprocess
import sys

def run_quarto_render():
    """
    Executes the Quarto render command for the BOCOM BBM report
    and streams its output to the console in real-time.
    """
    command = [
        "quarto",
        "render",
        "bocom_bbm_report.qmd",
        "--to", "pdf",
        "-P", "ref_bacen:513438999",
        "-P", "start_date:'2023-01-01'",
        "-P", "end_date:'2023-12-31'"
    ]

    print(f"Executing command: {' '.join(command)}", file=sys.stderr)

    try:
        # Use Popen to capture and stream output in real-time.
        # Stderr is redirected to stdout to capture all output in one stream.
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1  # Line-buffered
        ) as process:
            # Read and print output line by line as it comes in.
            for line in process.stdout:
                print(line, end='')

        if process.returncode != 0:
            print(f"\nError: Quarto render failed with exit code {process.returncode}.", file=sys.stderr)
            sys.exit(process.returncode)

    except FileNotFoundError:
        print("\nError: 'quarto' command not found. Make sure Quarto CLI is installed and in your system's PATH.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_quarto_render()