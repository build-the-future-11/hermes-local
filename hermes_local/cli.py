import argparse
import json

import httpx

from hermes_local.backend import OllamaClient
from hermes_local.chat import HermesChat


def main() -> int:
    parser = argparse.ArgumentParser(description="Experimental Hermes powered by your chosen base.")
    parser.add_argument("command", choices=["chat", "doctor"])
    parser.add_argument("--model")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.command == "chat" and not args.model:
        parser.error("chat requires --model with an exact installed model name")
    try:
        client = OllamaClient(base_url=args.base_url,
                              timeout_seconds=5 if args.command == "doctor" else 120)
        if args.command == "doctor":
            version = client.version()
            models = client.list_models()
            digest = client.model_digest(args.model) if args.model else None
            print(json.dumps({"status": "AVAILABLE_UNQUALIFIED" if models else "NO_MODELS",
                              "backend_version": version, "models": models,
                              "digest": digest, "generation_verified": False}))
            return 0 if models else 1
        session = HermesChat(client, args.model, context_tokens=args.context_tokens,
                             max_tokens=args.max_tokens)
    except (httpx.HTTPError, ValueError):
        print("Setup failed. Start Ollama, check 'ollama list', model name and context settings.")
        return 1
    print(f"Experimental Hermes powered by {args.model}\nDigest: {session.digest}")
    print("Session only; /clear or /exit. Output is unverified and provisional until completion.")
    while True:
        try:
            prompt = input("You> ")
            if prompt.strip() == "/exit":
                return 0
            if prompt.strip() == "/clear":
                session.clear()
                print("Conversation cleared.")
                continue
            print("Hermes> ", end="", flush=True)
            result = session.answer(prompt, on_token=lambda text: print(text, end="", flush=True))
            print()
            if session.truncated:
                print("[Older turns omitted to fit context.]")
            if result.finish_reason == "length":
                print("[Output limit reached; answer may be incomplete.]")
        except (KeyboardInterrupt, EOFError):
            print("\nSession closed.")
            return 0
        except (httpx.HTTPError, ValueError):
            print("\nGeneration failed. Discard partial output; failed turn was not saved.")


if __name__ == "__main__":
    raise SystemExit(main())
