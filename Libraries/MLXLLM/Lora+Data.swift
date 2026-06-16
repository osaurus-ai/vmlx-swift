// Copyright © 2024 Apple Inc.

import Foundation

enum LoRADataError: LocalizedError {
    case fileNotFound(URL, String)
    case unsupportedFileType(URL)

    var errorDescription: String? {
        switch self {
        case .fileNotFound(let directory, let name):
            return String(
                localized: "Could not find data file '\(name)' in directory '\(directory.path())'.")
        case .unsupportedFileType(let url):
            return String(
                localized: "Unsupported LoRA data file type for '\(url.lastPathComponent)'. Use a .jsonl or .txt file.")
        }
    }
}

/// Load a LoRA data file.
///
/// Given a directory and a base name, e.g. `train`, this will load a `.jsonl` or `.txt` file
/// if possible.
public func loadLoRAData(directory: URL, name: String) throws -> [String] {
    let extensions = ["jsonl", "txt"]

    for ext in extensions {
        let url = directory.appending(component: "\(name).\(ext)")
        if FileManager.default.fileExists(atPath: url.path()) {
            return try loadLoRAData(url: url)
        }
    }

    throw LoRADataError.fileNotFound(directory, name)
}

/// Load a .txt or .jsonl file and return the contents
public func loadLoRAData(url: URL) throws -> [String] {
    switch url.pathExtension {
    case "jsonl":
        return try loadJSONL(url: url)

    case "txt":
        return try loadLines(url: url)

    default:
        throw LoRADataError.unsupportedFileType(url)

    }
}

func loadJSONL(url: URL) throws -> [String] {

    struct Line: Codable {
        let text: String?
    }

    return try String(contentsOf: url)
        .components(separatedBy: .newlines)
        .filter {
            $0.first == "{"
        }
        .compactMap {
            try JSONDecoder().decode(Line.self, from: $0.data(using: .utf8)!).text
        }
}

func loadLines(url: URL) throws -> [String] {
    try String(contentsOf: url)
        .components(separatedBy: .newlines)
        .filter { !$0.isEmpty }
}
