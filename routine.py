if __name__ == '__main__':
    import subprocess
    import sys
    import os
    from azure.storage.blob import BlobServiceClient

    def upa_file_blob(local_file_path: str, blob_container: str, blob_path: str, delete_original_file: bool = False):
        """Uploads a file to Azure Blob Storage at the specified blob path."""
        connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "DefaultEndpointsProtocol=https;AccountName=pesquanta94c;AccountKey=MXdP1yoZ0KMzO471kyIHWHdlkxdVytpe+ExVsLbyZ9mBJcqAR5X2b3u+emLFJghYkc3Yc3ltDqHc+ASt3ACcZA==;EndpointSuffix=core.windows.net")
        container_name = blob_container
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)
        
        for attempt in range(3):
            try:
                if blob_client.exists():
                    print(f"Blob {blob_path} already exists. Deleting it before upload.", file=sys.stderr)
                    blob_client.delete_blob()
                
                with open(local_file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                    print(f"[Sucesso] O arquivo {local_file_path} foi upado para o blob em {container_name}/{blob_path}", file=sys.stderr)
                    if delete_original_file:
                        os.remove(local_file_path)
                        print(f"Arquivo local {local_file_path} deletado.", file=sys.stderr)
                    return  # Success, exit the function
            except Exception as e:
                print(f"Falha ao upar o arquivo {local_file_path} (tentativa {attempt + 1}/3). Erro: {e}", file=sys.stderr)
                if attempt < 2:
                    time.sleep(5)
        
        print(f"Falha final ao upar o arquivo {local_file_path} após 3 tentativas.", file=sys.stderr)
        sys.exit(1)

    ######## Função para baixar um arquivo do blob
    def baixa_file_blob(blob_container: str, blob_path: str, local_file_path: str):
        """Downloads a file from Azure Blob Storage to the local directory."""
        connect_str = os.getenv(
            "AZURE_STORAGE_CONNECTION_STRING",
            "DefaultEndpointsProtocol=https;AccountName=pesquanta94c;AccountKey=MXdP1yoZ0KMzO471kyIHWHdlkxdVytpe+ExVsLbyZ9mBJcqAR5X2b3u+emLFJghYkc3Yc3ltDqHc+ASt3ACcZA==;EndpointSuffix=core.windows.net"
        )
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        blob_client = blob_service_client.get_blob_client(container=blob_container, blob=blob_path)

        for attempt in range(3):
            try:
                if blob_client.exists():
                    with open(local_file_path, "wb") as download_file:
                        download_stream = blob_client.download_blob()
                        download_file.write(download_stream.readall())
                        print(f"[Sucesso] O arquivo {local_file_path} foi baixado para o diretório local", file=sys.stderr)
                        return  # Success, exit the function
                else:
                    print(f"[Falha] O arquivo {blob_path} não existe no blob {blob_container}", file=sys.stderr)
                    sys.exit(1)
            except Exception as e:
                print(f"[Falha] Erro ao baixar {blob_path} (tentativa {attempt + 1}/3). Erro: {e}", file=sys.stderr)
                if attempt < 2:
                    time.sleep(5)

        print(f"Falha final ao baixar o arquivo {blob_path} após 3 tentativas.", file=sys.stderr)
        sys.exit(1)





    def run_quarto_render():
        """
        Accepts two arguments: input blob file path and output blob file path (including container and blob path).
        Downloads the input XML, infers refbacen and group_name from the filename, runs Quarto, and uploads the PDF to the output blob path.
        Usage: python routine.py <input_blob_uri> <output_blob_uri>
        """
        if len(sys.argv) < 3:
            print("Usage: python routine.py <input_blob_uri> <output_blob_uri>", file=sys.stderr)
            sys.exit(2)

        # Parse input and output blob URIs (format: azure://container/blob_path)
        def parse_azure_uri(uri):
            if not uri.startswith("azure://"):
                raise ValueError(f"Invalid Azure URI: {uri}")
            _, rest = uri.split("azure://", 1)
            container, blob_path = rest.split("/", 1)
            return container, blob_path

        input_blob_uri = sys.argv[1]
        output_blob_uri = sys.argv[2]
        input_container, input_blob_path = parse_azure_uri(input_blob_uri)
        output_container, output_blob_path = parse_azure_uri(output_blob_uri)

        # Download the XML from blob to local file (use the filename part only)
        local_xml_filename = os.path.basename(input_blob_path)
        print(f"Downloading {input_blob_path} from container {input_container}...", file=sys.stderr)
        baixa_file_blob(input_container, input_blob_path, local_xml_filename)

        # Infer refbacen and group_name from filename: <refbacen>_<groupname>.xml
        stem = os.path.splitext(local_xml_filename)[0]
        if "_" not in stem:
            print(f"Error: filename '{local_xml_filename}' does not follow the expected '<refbacen>_<groupname>.xml' format.", file=sys.stderr)
            sys.exit(3)
        refbacen, group_name = stem.split("_", 1)

        # prepare Quarto command
        input_filename = "bocom_bbm_report.qmd"
        desired_output_filename = f"{stem}.pdf"
        generated_output_filename = f"{os.path.splitext(input_filename)[0]}.pdf"

        command = [
            "uv",
            "run",
            "quarto",
            "render",
            input_filename,
            "--to", "PrettyPDF-pdf",
            "-P", f"ref_bacen:{refbacen}",
            "-P", f"produtor:'{group_name}'"
        ]
        print(f"Executing command: {' '.join(command)}", file=sys.stderr)

        try:
            with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1
            ) as process:
                for line in process.stdout:
                    print(line, end='')

            if process.returncode != 0:
                print(f"\nError: Quarto render failed with exit code {process.returncode}.", file=sys.stderr)
                sys.exit(process.returncode)

            print("\nQuarto render completed successfully.", file=sys.stderr)

            # If Quarto produced the default PDF, rename it to match the XML basename
            if os.path.exists(generated_output_filename):
                try:
                    if os.path.exists(desired_output_filename):
                        os.remove(desired_output_filename)
                    os.replace(generated_output_filename, desired_output_filename)
                    output_filename = desired_output_filename
                    print(f"Renamed {generated_output_filename} -> {output_filename}", file=sys.stderr)
                except Exception as e:
                    print(f"Error renaming output file: {e}", file=sys.stderr)
                    sys.exit(1)
            else:
                if os.path.exists(desired_output_filename):
                    output_filename = desired_output_filename
                else:
                    print(f"Error: Output file not found after render (checked {generated_output_filename} and {desired_output_filename}).", file=sys.stderr)
                    sys.exit(1)

            # Upload the generated PDF to Azure Blob Storage at the specified output blob path
            if os.path.exists(output_filename):
                print(f"Uploading {output_filename} to Azure Blob Storage at {output_container}/{output_blob_path}...", file=sys.stderr)
                upa_file_blob(
                    local_file_path=output_filename,
                    blob_container=output_container,
                    blob_path=output_blob_path,
                    delete_original_file=False
                )

        except FileNotFoundError:
            print("\nError: 'quarto' command not found. Make sure Quarto CLI is installed and in your system's PATH.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)

    if __name__ == "__main__":
        run_quarto_render()
