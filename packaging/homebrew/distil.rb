# Homebrew formula for Distil (PyPI: distil-llm).
#
# CANONICAL COPY: https://github.com/dshakes/homebrew-tap (Formula/distil.rb).
# CI's `bump-homebrew` release job rewrites the tap on every v* tag, so the tap
# is always current; this in-repo copy is a reference snapshot and may lag.
#
# To install:
#   brew tap dshakes/tap
#   brew install dshakes/tap/distil
#
# sha256 is for the v1.11.2 source tarball. To recompute for a new version:
#   curl -sL https://github.com/dshakes/distil/archive/refs/tags/vX.Y.Z.tar.gz | shasum -a 256

class Distil < Formula
  desc "Compression with a quality contract — context compression for LLM agentic runtimes"
  homepage "https://github.com/dshakes/distil"
  url "https://github.com/dshakes/distil/archive/refs/tags/v1.22.0.tar.gz"
  sha256 "297be78cb09bde589c1f970a2dbf77e5d199250b958543759dfe947c3032a9c8"
  license "Apache-2.0"
  version "1.22.0"

  depends_on "python@3.12"

  def install
    # Create an isolated venv in libexec so Distil's stdlib-only package does
    # not pollute the user's Python environment.
    venv = libexec/"venv"
    system "python3.12", "-m", "venv", venv
    system "#{venv}/bin/pip", "install", "--no-deps", "."

    # Expose the `distil` entry-point as a shim in bin/.
    (bin/"distil").write <<~SH
      #!/bin/sh
      exec "#{venv}/bin/distil" "$@"
    SH
    chmod 0755, bin/"distil"
  end

  test do
    # Smoke-test: --version must exit 0 and print a version string.
    system bin/"distil", "--version"
  end
end
