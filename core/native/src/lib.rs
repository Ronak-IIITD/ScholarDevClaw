//! ScholarDevClaw native Rust extension — fast multi-language AST walking.
//!
//! Replaces the Python `_walk_for_elements_and_imports` hot path (~66% of analyze() time)
//! with a zero-copy Rust implementation using tree-sitter directly.

use pyo3::prelude::*;

// ─── Data structs exposed to Python ──────────────────────────────────────────

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct PyCodeElement {
    #[pyo3(get)]
    pub elem_type: String,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub file: String,
    #[pyo3(get)]
    pub line: usize,
    #[pyo3(get)]
    pub end_line: usize,
    #[pyo3(get)]
    pub language: String,
    #[pyo3(get)]
    pub visibility: String,
    #[pyo3(get)]
    pub parameters: Vec<String>,
    #[pyo3(get)]
    pub return_type: String,
    #[pyo3(get)]
    pub decorators: Vec<String>,
    #[pyo3(get)]
    pub parent_class: Option<String>,
    #[pyo3(get)]
    pub dependencies: Vec<String>,
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct PyImportStatement {
    #[pyo3(get)]
    pub module: String,
    #[pyo3(get)]
    pub names: Vec<String>,
    #[pyo3(get)]
    pub file: String,
    #[pyo3(get)]
    pub line: usize,
    #[pyo3(get)]
    pub is_from: bool,
    #[pyo3(get)]
    pub alias: Option<String>,
}

// ─── Walk result ─────────────────────────────────────────────────────────────

#[pyclass]
pub struct WalkResult {
    #[pyo3(get)]
    pub elements: Vec<PyCodeElement>,
    #[pyo3(get)]
    pub imports: Vec<PyImportStatement>,
}

// ─── Tree-sitter language registry ───────────────────────────────────────────

fn get_language(name: &str) -> Option<tree_sitter::Language> {
    match name {
        "python" => Some(tree_sitter_python::language()),
        "javascript" => Some(tree_sitter_javascript::language()),
        "typescript" => Some(tree_sitter_typescript::language_typescript()),
        "go" => Some(tree_sitter_go::language()),
        "rust" => Some(tree_sitter_rust::language()),
        "java" => Some(tree_sitter_java::language()),
        _ => None,
    }
}

// ─── Helper: node text ───────────────────────────────────────────────────────

#[inline(always)]
fn node_text<'a>(node: tree_sitter::Node, source: &'a [u8]) -> &'a str {
    &std::str::from_utf8(&source[node.start_byte()..node.end_byte()])
        .unwrap_or("")
}

#[inline(always)]
fn child_by_field<'a>(
    node: tree_sitter::Node<'a>,
    field: &str,
    _source: &[u8],
) -> Option<tree_sitter::Node<'a>> {
    node.child_by_field_name(field)
}

#[inline(always)]
fn extract_string_content(node: tree_sitter::Node, source: &[u8]) -> String {
    let text = node_text(node, source);
    if text.len() >= 2
        && (text.starts_with('"') || text.starts_with('\'') || text.starts_with('`'))
        && (text.ends_with('"') || text.ends_with('\'') || text.ends_with('`'))
    {
        return text[1..text.len() - 1].to_string();
    }
    // Check for string_content / string_fragment children
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "string_fragment" || child.kind() == "string_content" {
            return node_text(child, source).to_string();
        }
    }
    text.to_string()
}

// ─── Main walk function ──────────────────────────────────────────────────────

fn walk_ast(
    node: tree_sitter::Node,
    rel_path: &str,
    language: &str,
    source: &[u8],
    parent_class: Option<&str>,
    elements: &mut Vec<PyCodeElement>,
    imports: &mut Vec<PyImportStatement>,
) {
    // Extract elements
    extract_element(node, rel_path, language, source, parent_class, elements);

    // Extract imports
    extract_import(node, rel_path, language, source, imports);

    // Compute new parent_class for children
    let mut new_parent: Option<String> = parent_class.map(|s| s.to_string());

    match language {
        "python" if node.kind() == "class_definition" => {
            if let Some(name_node) = child_by_field(node, "name", source) {
                new_parent = Some(node_text(name_node, source).to_string());
            }
        }
        "javascript" | "typescript"
            if node.kind() == "class_declaration" || node.kind() == "class" =>
        {
            if let Some(name_node) = child_by_field(node, "name", source) {
                new_parent = Some(node_text(name_node, source).to_string());
            }
        }
        "java" if node.kind() == "class_declaration" => {
            if let Some(name_node) = child_by_field(node, "name", source) {
                new_parent = Some(node_text(name_node, source).to_string());
            }
        }
        "rust" if node.kind() == "impl_item" => {
            if let Some(type_node) = child_by_field(node, "type", source) {
                new_parent = Some(node_text(type_node, source).to_string());
            }
        }
        _ => {}
    }

    // Recurse
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        walk_ast(
            child,
            rel_path,
            language,
            source,
            new_parent.as_deref(),
            elements,
            imports,
        );
    }
}

// ─── Element extraction per language ─────────────────────────────────────────

fn extract_element(
    node: tree_sitter::Node,
    rel_path: &str,
    language: &str,
    source: &[u8],
    parent_class: Option<&str>,
    elements: &mut Vec<PyCodeElement>,
) {
    match language {
        "python" => extract_python_element(node, rel_path, source, parent_class, elements),
        "javascript" => extract_js_element(node, rel_path, language, source, parent_class, elements),
        "typescript" => extract_ts_element(node, rel_path, source, parent_class, elements),
        "go" => extract_go_element(node, rel_path, source, parent_class, elements),
        "rust" => extract_rust_element(node, rel_path, source, parent_class, elements),
        "java" => extract_java_element(node, rel_path, source, parent_class, elements),
        _ => {}
    }
}

// ─── Python ──────────────────────────────────────────────────────────────────

fn extract_python_element(
    node: tree_sitter::Node,
    rel_path: &str,
    source: &[u8],
    parent_class: Option<&str>,
    elements: &mut Vec<PyCodeElement>,
) {
    match node.kind() {
        "class_definition" => {
            let name_node = child_by_field(node, "name", source);
            if name_node.is_none() {
                return;
            }
            let name = node_text(name_node.unwrap(), source);
            let decorators = get_python_decorators(node, source);
            let bases = get_python_bases(node, source);

            elements.push(PyCodeElement {
                elem_type: "class".to_string(),
                name: name.to_string(),
                file: rel_path.to_string(),
                line: node.start_position().row + 1,
                end_line: node.end_position().row + 1,
                language: "python".to_string(),
                visibility: if name.starts_with('_') {
                    "private".to_string()
                } else {
                    "public".to_string()
                },
                decorators,
                dependencies: bases,
                ..Default::default()
            });
        }
        "function_definition" => {
            let name_node = child_by_field(node, "name", source);
            if name_node.is_none() {
                return;
            }
            let name = node_text(name_node.unwrap(), source);
            let is_async = has_child_of_type(node, "async");
            let decorators = get_python_decorators(node, source);
            let params = get_python_params(node, source);
            let return_type = get_python_return_type(node, source);

            let visibility = if name.starts_with("__") && name.ends_with("__") {
                "public".to_string()
            } else if name.starts_with("__") {
                "private".to_string()
            } else if name.starts_with('_') {
                "protected".to_string()
            } else {
                "public".to_string()
            };

            let elem_type = if parent_class.is_some() {
                if is_async {
                    "async_method"
                } else {
                    "method"
                }
            } else if is_async {
                "async_function"
            } else {
                "function"
            };

            elements.push(PyCodeElement {
                elem_type: elem_type.to_string(),
                name: name.to_string(),
                file: rel_path.to_string(),
                line: node.start_position().row + 1,
                end_line: node.end_position().row + 1,
                language: "python".to_string(),
                visibility,
                parameters: params,
                return_type,
                decorators,
                parent_class: parent_class.map(|s| s.to_string()),
                ..Default::default()
            });
        }
        _ => {}
    }
}

fn get_python_decorators(node: tree_sitter::Node, source: &[u8]) -> Vec<String> {
    let mut decorators = Vec::new();
    if let Some(parent) = node.parent() {
        if parent.kind() == "decorated_definition" {
            let mut cursor = parent.walk();
            for child in parent.named_children(&mut cursor) {
                if child.kind() == "decorator" {
                    let mut parts = Vec::new();
                    let mut dcursor = child.walk();
                    for dc in child.named_children(&mut dcursor) {
                        if dc.kind() != "@" {
                            parts.push(node_text(dc, source));
                        }
                    }
                    if !parts.is_empty() {
                        decorators.push(parts.join(""));
                    }
                }
            }
        }
    }
    decorators
}

fn get_python_bases(node: tree_sitter::Node, source: &[u8]) -> Vec<String> {
    let mut bases = Vec::new();
    if let Some(superclasses) = child_by_field(node, "superclasses", source) {
        let mut cursor = superclasses.walk();
        for child in superclasses.named_children(&mut cursor) {
            if child.kind() == "identifier" || child.kind() == "attribute" {
                bases.push(node_text(child, source).to_string());
            }
        }
    } else {
        // Fallback: check for argument_list
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            if child.kind() == "argument_list" {
                let mut acursor = child.walk();
                for arg in child.named_children(&mut acursor) {
                    if arg.kind() == "identifier" || arg.kind() == "attribute" {
                        bases.push(node_text(arg, source).to_string());
                    }
                }
            }
        }
    }
    bases
}

fn get_python_params(node: tree_sitter::Node, source: &[u8]) -> Vec<String> {
    let mut params = Vec::new();
    if let Some(params_node) = child_by_field(node, "parameters", source) {
        let mut cursor = params_node.walk();
        for child in params_node.named_children(&mut cursor) {
            match child.kind() {
                "identifier"
                | "typed_parameter"
                | "default_parameter"
                | "typed_default_parameter"
                | "list_splat_pattern"
                | "dictionary_splat_pattern" => {
                    params.push(node_text(child, source).to_string());
                }
                _ => {}
            }
        }
    }
    params
}

fn get_python_return_type(node: tree_sitter::Node, source: &[u8]) -> String {
    if let Some(return_node) = child_by_field(node, "return_type", source) {
        return node_text(return_node, source).to_string();
    }
    // Fallback: look for -> pattern
    let mut cursor = node.walk();
    let mut found_arrow = false;
    for child in node.named_children(&mut cursor) {
        if found_arrow && child.kind() == "type" {
            return node_text(child, source).to_string();
        }
        if node_text(child, source) == "->" {
            found_arrow = true;
        }
    }
    String::new()
}

// ─── JavaScript ──────────────────────────────────────────────────────────────

fn extract_js_element(
    node: tree_sitter::Node,
    rel_path: &str,
    language: &str,
    source: &[u8],
    parent_class: Option<&str>,
    elements: &mut Vec<PyCodeElement>,
) {
    match node.kind() {
        "class_declaration" => {
            let name_node = child_by_field(node, "name", source);
            if name_node.is_none() {
                return;
            }
            let name = node_text(name_node.unwrap(), source);
            let bases = get_js_heritage(node, source);

            elements.push(PyCodeElement {
                elem_type: "class".to_string(),
                name: name.to_string(),
                file: rel_path.to_string(),
                line: node.start_position().row + 1,
                end_line: node.end_position().row + 1,
                language: language.to_string(),
                dependencies: bases,
                ..Default::default()
            });
        }
        "function_declaration" => {
            let name_node = child_by_field(node, "name", source);
            if name_node.is_none() {
                return;
            }
            let name = node_text(name_node.unwrap(), source);
            let is_async = has_child_of_type(node, "async");
            let params = get_js_params(node, source);
            let return_type = if language == "typescript" {
                get_ts_return_type(node, source)
            } else {
                String::new()
            };

            elements.push(PyCodeElement {
                elem_type: if is_async {
                    "async_function".to_string()
                } else {
                    "function".to_string()
                },
                name: name.to_string(),
                file: rel_path.to_string(),
                line: node.start_position().row + 1,
                end_line: node.end_position().row + 1,
                language: language.to_string(),
                parameters: params,
                return_type,
                ..Default::default()
            });
        }
        "method_definition" => {
            let name_node = child_by_field(node, "name", source);
            if name_node.is_none() {
                return;
            }
            let name = node_text(name_node.unwrap(), source);
            let is_async = has_child_of_type(node, "async");
            let is_getter = has_child_of_type(node, "get");
            let is_setter = has_child_of_type(node, "set");
            let params = get_js_params(node, source);
            let return_type = if language == "typescript" {
                get_ts_return_type(node, source)
            } else {
                String::new()
            };

            let elem_type = if is_getter {
                "getter".to_string()
            } else if is_setter {
                "setter".to_string()
            } else if is_async {
                "async_method".to_string()
            } else {
                "method".to_string()
            };

            let visibility = if name.starts_with('#') {
                "private".to_string()
            } else {
                "public".to_string()
            };

            elements.push(PyCodeElement {
                elem_type,
                name: name.to_string(),
                file: rel_path.to_string(),
                line: node.start_position().row + 1,
                end_line: node.end_position().row + 1,
                language: language.to_string(),
                visibility,
                parameters: params,
                return_type,
                parent_class: parent_class.map(|s| s.to_string()),
                ..Default::default()
            });
        }
        "lexical_declaration" | "variable_declaration" => {
            extract_js_arrow_functions(node, rel_path, language, source, elements);
        }
        _ => {}
    }
}

fn get_js_heritage(node: tree_sitter::Node, source: &[u8]) -> Vec<String> {
    let mut bases = Vec::new();
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "class_heritage" {
            let mut hcursor = child.walk();
            for hc in child.named_children(&mut hcursor) {
                if hc.kind() == "identifier" {
                    bases.push(node_text(hc, source).to_string());
                }
            }
        }
    }
    bases
}

fn get_js_params(node: tree_sitter::Node, source: &[u8]) -> Vec<String> {
    let mut params = Vec::new();
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "formal_parameters" {
            let mut pcursor = child.walk();
            for p in child.named_children(&mut pcursor) {
                match p.kind() {
                    "identifier"
                    | "required_parameter"
                    | "optional_parameter"
                    | "rest_pattern"
                    | "assignment_pattern" => {
                        params.push(node_text(p, source).to_string());
                    }
                    _ => {}
                }
            }
            break;
        }
    }
    params
}

fn get_ts_return_type(node: tree_sitter::Node, source: &[u8]) -> String {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "type_annotation" {
            let mut tcursor = child.walk();
            for tc in child.named_children(&mut tcursor) {
                if tc.kind() != ":" {
                    return node_text(tc, source).to_string();
                }
            }
        }
    }
    String::new()
}

fn extract_js_arrow_functions(
    node: tree_sitter::Node,
    rel_path: &str,
    language: &str,
    source: &[u8],
    elements: &mut Vec<PyCodeElement>,
) {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "variable_declarator" {
            let name_node = child_by_field(child, "name", source);
            let value_node = child_by_field(child, "value", source);
            if let (Some(name_n), Some(val_n)) = (name_node, value_node) {
                if val_n.kind() == "arrow_function" {
                    let name = node_text(name_n, source);
                    let is_async = has_child_of_type(val_n, "async");
                    let params = get_js_params(val_n, source);
                    let return_type = if language == "typescript" {
                        get_ts_return_type(val_n, source)
                    } else {
                        String::new()
                    };

                    elements.push(PyCodeElement {
                        elem_type: if is_async {
                            "async_function".to_string()
                        } else {
                            "function".to_string()
                        },
                        name: name.to_string(),
                        file: rel_path.to_string(),
                        line: val_n.start_position().row + 1,
                        end_line: val_n.end_position().row + 1,
                        language: language.to_string(),
                        parameters: params,
                        return_type,
                        ..Default::default()
                    });
                }
            }
        }
    }
}

// ─── TypeScript-specific ─────────────────────────────────────────────────────

fn extract_ts_element(
    node: tree_sitter::Node,
    rel_path: &str,
    source: &[u8],
    parent_class: Option<&str>,
    elements: &mut Vec<PyCodeElement>,
) {
    match node.kind() {
        "interface_declaration" => {
            if let Some(name_node) = child_by_field(node, "name", source) {
                elements.push(PyCodeElement {
                    elem_type: "interface".to_string(),
                    name: node_text(name_node, source).to_string(),
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                    language: "typescript".to_string(),
                    ..Default::default()
                });
            }
        }
        "type_alias_declaration" => {
            if let Some(name_node) = child_by_field(node, "name", source) {
                elements.push(PyCodeElement {
                    elem_type: "type_alias".to_string(),
                    name: node_text(name_node, source).to_string(),
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                    language: "typescript".to_string(),
                    ..Default::default()
                });
            }
        }
        "enum_declaration" => {
            if let Some(name_node) = child_by_field(node, "name", source) {
                elements.push(PyCodeElement {
                    elem_type: "enum".to_string(),
                    name: node_text(name_node, source).to_string(),
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                    language: "typescript".to_string(),
                    ..Default::default()
                });
            }
        }
        _ => {
            // Delegate common JS/TS nodes to JS handler
            extract_js_element(
                node,
                rel_path,
                "typescript",
                source,
                parent_class,
                elements,
            );
        }
    }
}

// ─── Go ──────────────────────────────────────────────────────────────────────

fn extract_go_element(
    node: tree_sitter::Node,
    rel_path: &str,
    source: &[u8],
    parent_class: Option<&str>,
    elements: &mut Vec<PyCodeElement>,
) {
    let _ = parent_class;
    match node.kind() {
        "function_declaration" => {
            if let Some(name_node) = child_by_field(node, "name", source) {
                let name = node_text(name_node, source);
                let params = get_go_params(node, source);
                let return_type = get_go_return_type(node, source);
                let visibility = if name.starts_with(|c: char| c.is_uppercase()) {
                    "public".to_string()
                } else {
                    "private".to_string()
                };

                elements.push(PyCodeElement {
                    elem_type: "function".to_string(),
                    name: name.to_string(),
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                    language: "go".to_string(),
                    visibility,
                    parameters: params,
                    return_type,
                    ..Default::default()
                });
            }
        }
        "method_declaration" => {
            if let Some(name_node) = child_by_field(node, "name", source) {
                let name = node_text(name_node, source);
                let params = get_go_params(node, source);
                let return_type = get_go_return_type(node, source);
                let visibility = if name.starts_with(|c: char| c.is_uppercase()) {
                    "public".to_string()
                } else {
                    "private".to_string()
                };

                // Extract receiver
                let mut receiver = None;
                let mut cursor = node.walk();
                for child in node.named_children(&mut cursor) {
                    if child.kind() == "parameter_list" {
                        let mut pcursor = child.walk();
                        for pc in child.named_children(&mut pcursor) {
                            if pc.kind() == "parameter_declaration" {
                                let mut tcursor = pc.walk();
                                for tc in pc.named_children(&mut tcursor) {
                                    if tc.kind() == "type_identifier" || tc.kind() == "pointer_type"
                                    {
                                        let mut txt = node_text(tc, source).to_string();
                                        if txt.starts_with('*') {
                                            txt.remove(0);
                                        }
                                        receiver = Some(txt);
                                    }
                                }
                            }
                        }
                        break;
                    }
                }

                elements.push(PyCodeElement {
                    elem_type: "method".to_string(),
                    name: name.to_string(),
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                    language: "go".to_string(),
                    visibility,
                    parameters: params,
                    return_type,
                    parent_class: receiver,
                    ..Default::default()
                });
            }
        }
        "type_declaration" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if child.kind() == "type_spec" {
                    if let Some(name_node) = child_by_field(child, "name", source) {
                        let name = node_text(name_node, source);
                        let visibility = if name.starts_with(|c: char| c.is_uppercase()) {
                            "public".to_string()
                        } else {
                            "private".to_string()
                        };
                        let type_node = child_by_field(child, "type", source);
                        let elem_type = if let Some(tn) = type_node {
                            match tn.kind() {
                                "interface_type" => "interface",
                                "struct_type" => "struct",
                                _ => "type_alias",
                            }
                        } else {
                            "type_alias"
                        };

                        elements.push(PyCodeElement {
                            elem_type: elem_type.to_string(),
                            name: name.to_string(),
                            file: rel_path.to_string(),
                            line: child.start_position().row + 1,
                            end_line: child.end_position().row + 1,
                            language: "go".to_string(),
                            visibility,
                            ..Default::default()
                        });
                    }
                }
            }
        }
        _ => {}
    }
}

fn get_go_params(node: tree_sitter::Node, source: &[u8]) -> Vec<String> {
    let mut params = Vec::new();
    let mut param_lists_seen = 0u32;
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "parameter_list" {
            param_lists_seen += 1;
            // For method_declaration, first param_list is receiver
            if node.kind() == "method_declaration" && param_lists_seen == 1 {
                continue;
            }
            let mut pcursor = child.walk();
            for pc in child.named_children(&mut pcursor) {
                if pc.kind() == "parameter_declaration" {
                    params.push(node_text(pc, source).to_string());
                }
            }
            if node.kind() != "method_declaration" {
                break;
            }
        }
    }
    params
}

fn get_go_return_type(node: tree_sitter::Node, source: &[u8]) -> String {
    if let Some(result) = child_by_field(node, "result", source) {
        return node_text(result, source).to_string();
    }
    String::new()
}

// ─── Rust ────────────────────────────────────────────────────────────────────

fn extract_rust_element(
    node: tree_sitter::Node,
    rel_path: &str,
    source: &[u8],
    parent_class: Option<&str>,
    elements: &mut Vec<PyCodeElement>,
) {
    match node.kind() {
        "function_item" => {
            if let Some(name_node) = child_by_field(node, "name", source) {
                let name = node_text(name_node, source);
                let is_pub = has_child_of_type(node, "visibility_modifier");
                // In tree-sitter-rust, `async` is inside `function_modifiers` node
                let is_async = has_nested_keyword(node, "function_modifiers", "async");
                let params = get_rust_params(node, source);
                let return_type = get_rust_return_type(node, source);

                let elem_type = if parent_class.is_some() {
                    if is_async {
                        "async_method"
                    } else {
                        "method"
                    }
                } else if is_async {
                    "async_function"
                } else {
                    "function"
                };

                elements.push(PyCodeElement {
                    elem_type: elem_type.to_string(),
                    name: name.to_string(),
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                    language: "rust".to_string(),
                    visibility: if is_pub {
                        "public".to_string()
                    } else {
                        "private".to_string()
                    },
                    parameters: params,
                    return_type,
                    parent_class: parent_class.map(|s| s.to_string()),
                    ..Default::default()
                });
            }
        }
        "struct_item" => {
            if let Some(name_node) = child_by_field(node, "name", source) {
                let is_pub = has_child_of_type(node, "visibility_modifier");
                elements.push(PyCodeElement {
                    elem_type: "struct".to_string(),
                    name: node_text(name_node, source).to_string(),
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                    language: "rust".to_string(),
                    visibility: if is_pub {
                        "public".to_string()
                    } else {
                        "private".to_string()
                    },
                    ..Default::default()
                });
            }
        }
        "enum_item" => {
            if let Some(name_node) = child_by_field(node, "name", source) {
                let is_pub = has_child_of_type(node, "visibility_modifier");
                elements.push(PyCodeElement {
                    elem_type: "enum".to_string(),
                    name: node_text(name_node, source).to_string(),
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                    language: "rust".to_string(),
                    visibility: if is_pub {
                        "public".to_string()
                    } else {
                        "private".to_string()
                    },
                    ..Default::default()
                });
            }
        }
        "trait_item" => {
            if let Some(name_node) = child_by_field(node, "name", source) {
                let is_pub = has_child_of_type(node, "visibility_modifier");
                elements.push(PyCodeElement {
                    elem_type: "trait".to_string(),
                    name: node_text(name_node, source).to_string(),
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                    language: "rust".to_string(),
                    visibility: if is_pub {
                        "public".to_string()
                    } else {
                        "private".to_string()
                    },
                    ..Default::default()
                });
            }
        }
        "impl_item" => {
            if let Some(type_node) = child_by_field(node, "type", source) {
                let trait_node = child_by_field(node, "trait", source);
                let trait_name = trait_node.map(|tn| node_text(tn, source).to_string());
                let type_name = node_text(type_node, source).to_string();

                elements.push(PyCodeElement {
                    elem_type: "impl".to_string(),
                    name: type_name,
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                    language: "rust".to_string(),
                    dependencies: trait_name.map(|t| vec![t]).unwrap_or_default(),
                    ..Default::default()
                });
            }
        }
        _ => {}
    }
}

fn get_rust_params(node: tree_sitter::Node, source: &[u8]) -> Vec<String> {
    let mut params = Vec::new();
    if let Some(parameters) = child_by_field(node, "parameters", source) {
        let mut cursor = parameters.walk();
        for child in parameters.named_children(&mut cursor) {
            if child.kind() == "parameter" || child.kind() == "self_parameter" {
                params.push(node_text(child, source).to_string());
            }
        }
    }
    params
}

fn get_rust_return_type(node: tree_sitter::Node, source: &[u8]) -> String {
    if let Some(return_type) = child_by_field(node, "return_type", source) {
        return node_text(return_type, source).to_string();
    }
    String::new()
}

// ─── Java ────────────────────────────────────────────────────────────────────

fn extract_java_element(
    node: tree_sitter::Node,
    rel_path: &str,
    source: &[u8],
    parent_class: Option<&str>,
    elements: &mut Vec<PyCodeElement>,
) {
    let _ = parent_class;
    match node.kind() {
        "class_declaration" => {
            if let Some(name_node) = child_by_field(node, "name", source) {
                let name = node_text(name_node, source);
                let modifiers = get_java_modifiers(node, source);
                let visibility = java_visibility(&modifiers);

                let mut bases = Vec::new();
                if let Some(superclass) = child_by_field(node, "superclass", source) {
                    bases.push(node_text(superclass, source).to_string());
                }
                if let Some(interfaces) = child_by_field(node, "interfaces", source) {
                    let mut icursor = interfaces.walk();
                    for child in interfaces.named_children(&mut icursor) {
                        if child.kind() == "type_identifier" {
                            bases.push(node_text(child, source).to_string());
                        }
                    }
                }

                elements.push(PyCodeElement {
                    elem_type: "class".to_string(),
                    name: name.to_string(),
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                    language: "java".to_string(),
                    visibility,
                    dependencies: bases,
                    ..Default::default()
                });
            }
        }
        "interface_declaration" => {
            if let Some(name_node) = child_by_field(node, "name", source) {
                let modifiers = get_java_modifiers(node, source);
                elements.push(PyCodeElement {
                    elem_type: "interface".to_string(),
                    name: node_text(name_node, source).to_string(),
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                    language: "java".to_string(),
                    visibility: java_visibility(&modifiers),
                    ..Default::default()
                });
            }
        }
        "method_declaration" => {
            if let Some(name_node) = child_by_field(node, "name", source) {
                let name = node_text(name_node, source);
                let modifiers = get_java_modifiers(node, source);
                let params = get_java_params(node, source);
                let return_type = child_by_field(node, "type", source)
                    .map(|tn| node_text(tn, source).to_string())
                    .unwrap_or_default();

                elements.push(PyCodeElement {
                    elem_type: "method".to_string(),
                    name: name.to_string(),
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                    language: "java".to_string(),
                    visibility: java_visibility(&modifiers),
                    parameters: params,
                    return_type,
                    parent_class: parent_class.map(|s| s.to_string()),
                    ..Default::default()
                });
            }
        }
        "constructor_declaration" => {
            if let Some(name_node) = child_by_field(node, "name", source) {
                let modifiers = get_java_modifiers(node, source);
                let params = get_java_params(node, source);

                elements.push(PyCodeElement {
                    elem_type: "constructor".to_string(),
                    name: node_text(name_node, source).to_string(),
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                    language: "java".to_string(),
                    visibility: java_visibility(&modifiers),
                    parameters: params,
                    parent_class: parent_class.map(|s| s.to_string()),
                    ..Default::default()
                });
            }
        }
        "enum_declaration" => {
            if let Some(name_node) = child_by_field(node, "name", source) {
                let modifiers = get_java_modifiers(node, source);
                elements.push(PyCodeElement {
                    elem_type: "enum".to_string(),
                    name: node_text(name_node, source).to_string(),
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                    language: "java".to_string(),
                    visibility: java_visibility(&modifiers),
                    ..Default::default()
                });
            }
        }
        _ => {}
    }
}

fn get_java_modifiers(node: tree_sitter::Node, source: &[u8]) -> Vec<String> {
    let mut modifiers = Vec::new();
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "modifiers" {
            // Keywords like "public", "private" are anonymous tokens inside modifiers
            let mut mcursor = child.walk();
            for mod_node in child.children(&mut mcursor) {
                modifiers.push(node_text(mod_node, source).to_string());
            }
        }
    }
    modifiers
}

fn java_visibility(modifiers: &[String]) -> String {
    if modifiers.iter().any(|m| m == "public") {
        "public".to_string()
    } else if modifiers.iter().any(|m| m == "private") {
        "private".to_string()
    } else if modifiers.iter().any(|m| m == "protected") {
        "protected".to_string()
    } else {
        "package".to_string()
    }
}

fn get_java_params(node: tree_sitter::Node, source: &[u8]) -> Vec<String> {
    let mut params = Vec::new();
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "formal_parameters" {
            let mut pcursor = child.walk();
            for p in child.named_children(&mut pcursor) {
                if p.kind() == "formal_parameter" || p.kind() == "spread_parameter" {
                    params.push(node_text(p, source).to_string());
                }
            }
            break;
        }
    }
    params
}

// ─── Import extraction ───────────────────────────────────────────────────────

fn extract_import(
    node: tree_sitter::Node,
    rel_path: &str,
    language: &str,
    source: &[u8],
    imports: &mut Vec<PyImportStatement>,
) {
    match language {
        "python" => extract_python_import(node, rel_path, source, imports),
        "javascript" | "typescript" => extract_js_import(node, rel_path, source, imports),
        "go" => extract_go_import(node, rel_path, source, imports),
        "rust" => extract_rust_import(node, rel_path, source, imports),
        "java" => extract_java_import(node, rel_path, source, imports),
        _ => {}
    }
}

fn extract_python_import(
    node: tree_sitter::Node,
    rel_path: &str,
    source: &[u8],
    imports: &mut Vec<PyImportStatement>,
) {
    match node.kind() {
        "import_statement" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if child.kind() == "dotted_name" {
                    imports.push(PyImportStatement {
                        module: node_text(child, source).to_string(),
                        file: rel_path.to_string(),
                        line: node.start_position().row + 1,
                        ..Default::default()
                    });
                } else if child.kind() == "aliased_import" {
                    let mut dotted = None;
                    let mut alias = None;
                    let mut acursor = child.walk();
                    let mut first = true;
                    for ac in child.named_children(&mut acursor) {
                        if ac.kind() == "dotted_name" {
                            dotted = Some(node_text(ac, source).to_string());
                        } else if ac.kind() == "identifier" && !first {
                            alias = Some(node_text(ac, source).to_string());
                        }
                        first = false;
                    }
                    if let Some(d) = dotted {
                        imports.push(PyImportStatement {
                            module: d,
                            file: rel_path.to_string(),
                            line: node.start_position().row + 1,
                            alias,
                            ..Default::default()
                        });
                    }
                }
            }
        }
        "import_from_statement" => {
            let mut module_parts = Vec::new();
            let mut names = Vec::new();
            let mut found_import_keyword = false;

            // Iterate ALL children (including anonymous keywords like "from", "import")
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                match child.kind() {
                    "from" => continue,
                    "import" => {
                        found_import_keyword = true;
                        continue;
                    }
                    "relative_import" => {
                        let mut prefix = String::new();
                        let mut rcursor = child.walk();
                        for rc in child.named_children(&mut rcursor) {
                            if rc.kind() == "import_prefix" {
                                prefix.push_str(node_text(rc, source));
                            } else if rc.kind() == "dotted_name" {
                                prefix.push_str(node_text(rc, source));
                            }
                        }
                        module_parts.push(prefix);
                    }
                    "dotted_name" => {
                        if !found_import_keyword {
                            module_parts.push(node_text(child, source).to_string());
                        } else {
                            names.push(node_text(child, source).to_string());
                        }
                    }
                    "wildcard_import" => {
                        names.push("*".to_string());
                    }
                    "identifier" => {
                        if found_import_keyword {
                            names.push(node_text(child, source).to_string());
                        }
                    }
                    _ => {}
                }
            }

            let module = module_parts.join("");
            if !module.is_empty() || !names.is_empty() {
                imports.push(PyImportStatement {
                    module,
                    names,
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    is_from: true,
                    ..Default::default()
                });
            }
        }
        _ => {}
    }
}

fn extract_js_import(
    node: tree_sitter::Node,
    rel_path: &str,
    source: &[u8],
    imports: &mut Vec<PyImportStatement>,
) {
    match node.kind() {
        "import_statement" => {
            let mut module = String::new();
            let mut names = Vec::new();
            let mut alias = None;
            let mut cursor = node.walk();

            for child in node.named_children(&mut cursor) {
                if child.kind() == "string" {
                    module = extract_string_content(child, source);
                } else if child.kind() == "import_clause" {
                    let mut icursor = child.walk();
                    for ic in child.named_children(&mut icursor) {
                        match ic.kind() {
                            "named_imports" => {
                                let mut ncursor = ic.walk();
                                for spec in ic.named_children(&mut ncursor) {
                                    if spec.kind() == "import_specifier" {
                                        if let Some(name_n) = child_by_field(spec, "name", source) {
                                            names.push(node_text(name_n, source).to_string());
                                        } else {
                                            let mut scursor = spec.walk();
                                            for sc in spec.named_children(&mut scursor) {
                                                if sc.kind() == "identifier" {
                                                    names.push(
                                                        node_text(sc, source).to_string(),
                                                    );
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            "identifier" => {
                                names.push(node_text(ic, source).to_string());
                            }
                            "namespace_import" => {
                                let mut ncursor = ic.walk();
                                for nc in ic.named_children(&mut ncursor) {
                                    if nc.kind() == "identifier" {
                                        alias = Some(node_text(nc, source).to_string());
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            if !module.is_empty() {
                imports.push(PyImportStatement {
                    module,
                    names,
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    is_from: true,
                    alias,
                    ..Default::default()
                });
            }
        }
        "export_statement" => {
            let mut module = String::new();
            let mut names = Vec::new();
            let mut cursor = node.walk();

            for child in node.named_children(&mut cursor) {
                if child.kind() == "string" {
                    module = extract_string_content(child, source);
                } else if child.kind() == "export_clause" {
                    let mut ecursor = child.walk();
                    for spec in child.named_children(&mut ecursor) {
                        if spec.kind() == "export_specifier" {
                            let mut scursor = spec.walk();
                            for sc in spec.named_children(&mut scursor) {
                                if sc.kind() == "identifier" {
                                    names.push(node_text(sc, source).to_string());
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            if !module.is_empty() {
                imports.push(PyImportStatement {
                    module,
                    names,
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    is_from: true,
                    ..Default::default()
                });
            }
        }
        _ => {}
    }
}

fn extract_go_import(
    node: tree_sitter::Node,
    rel_path: &str,
    source: &[u8],
    imports: &mut Vec<PyImportStatement>,
) {
    if node.kind() == "import_declaration" {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "import_spec" => {
                    let path_node = child_by_field(child, "path", source);
                    let name_node = child_by_field(child, "name", source);
                    if let Some(pn) = path_node {
                        imports.push(PyImportStatement {
                            module: extract_string_content(pn, source),
                            file: rel_path.to_string(),
                            line: child.start_position().row + 1,
                            alias: name_node.map(|nn| node_text(nn, source).to_string()),
                            ..Default::default()
                        });
                    }
                }
                "import_spec_list" => {
                    let mut scursor = child.walk();
                    for spec in child.named_children(&mut scursor) {
                        if spec.kind() == "import_spec" {
                            let path_node = child_by_field(spec, "path", source);
                            let name_node = child_by_field(spec, "name", source);
                            if let Some(pn) = path_node {
                                imports.push(PyImportStatement {
                                    module: extract_string_content(pn, source),
                                    file: rel_path.to_string(),
                                    line: spec.start_position().row + 1,
                                    alias: name_node
                                        .map(|nn| node_text(nn, source).to_string()),
                                    ..Default::default()
                                });
                            }
                        }
                    }
                }
                "interpreted_string_literal" => {
                    imports.push(PyImportStatement {
                        module: extract_string_content(child, source),
                        file: rel_path.to_string(),
                        line: child.start_position().row + 1,
                        ..Default::default()
                    });
                }
                _ => {}
            }
        }
    }
}

fn extract_rust_import(
    node: tree_sitter::Node,
    rel_path: &str,
    source: &[u8],
    imports: &mut Vec<PyImportStatement>,
) {
    if node.kind() == "use_declaration" {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "use_as_clause"
                | "scoped_use_list"
                | "use_wildcard"
                | "scoped_identifier"
                | "identifier"
                | "use_list" => {
                    let path_text = node_text(child, source).to_string();
                    let (module, names) = if let Some(idx) = path_text.rfind("::") {
                        let module = path_text[..idx].to_string();
                        let name_part = &path_text[idx + 2..];
                        if name_part.starts_with('{') && name_part.ends_with('}') {
                            let names: Vec<String> = name_part[1..name_part.len() - 1]
                                .split(',')
                                .map(|n| n.trim().to_string())
                                .collect();
                            (module, names)
                        } else if name_part == "*" {
                            (module, vec!["*".to_string()])
                        } else {
                            (module, vec![name_part.to_string()])
                        }
                    } else {
                        (path_text, Vec::new())
                    };

                    imports.push(PyImportStatement {
                        module,
                        names,
                        file: rel_path.to_string(),
                        line: node.start_position().row + 1,
                        ..Default::default()
                    });
                    break;
                }
                _ => {}
            }
        }
    }
}

fn extract_java_import(
    node: tree_sitter::Node,
    rel_path: &str,
    source: &[u8],
    imports: &mut Vec<PyImportStatement>,
) {
    if node.kind() == "import_declaration" {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            if child.kind() == "scoped_identifier" {
                let full_path = node_text(child, source);
                let (module, name) = if let Some(idx) = full_path.rfind('.') {
                    (full_path[..idx].to_string(), full_path[idx + 1..].to_string())
                } else {
                    (full_path.to_string(), String::new())
                };

                let is_static = has_child_of_type(node, "static");
                let names = if name.is_empty() {
                    Vec::new()
                } else {
                    vec![name]
                };

                imports.push(PyImportStatement {
                    module,
                    names,
                    file: rel_path.to_string(),
                    line: node.start_position().row + 1,
                    is_from: is_static,
                    ..Default::default()
                });
                break;
            }
        }
    }
}

// ─── Utility helpers ─────────────────────────────────────────────────────────

#[inline(always)]
fn has_child_of_type(node: tree_sitter::Node, kind: &str) -> bool {
    let mut cursor = node.walk();
    // Check ALL children (including anonymous) — keywords like "async", "pub",
    // "get", "set" are anonymous tokens in tree-sitter grammars.
    for child in node.children(&mut cursor) {
        if child.kind() == kind {
            return true;
        }
    }
    false
}

/// Check for a keyword nested inside a wrapper node (e.g. `async` inside `function_modifiers`).
#[inline(always)]
fn has_nested_keyword(node: tree_sitter::Node, wrapper_kind: &str, keyword: &str) -> bool {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == wrapper_kind {
            let mut wcursor = child.walk();
            for grandchild in child.children(&mut wcursor) {
                if grandchild.kind() == keyword {
                    return true;
                }
            }
        }
    }
    false
}

// ─── Python module entry point ───────────────────────────────────────────────

/// Walk a single file's AST and return (elements, imports).
///
/// Usage from Python:
/// ```python
/// from scholardevclaw_native import walk_file
/// result = walk_file(source_bytes, rel_path, language)
/// # result.elements, result.imports
/// ```
#[pyfunction]
fn walk_file(
    _py: Python<'_>,
    source: Vec<u8>,
    rel_path: String,
    language: String,
) -> PyResult<WalkResult> {
    let lang = get_language(&language)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("Unsupported language: {language}")))?;

    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&lang).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to set language {language}: {e}"))
    })?;

    let tree = parser.parse(&source, None).ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to parse {rel_path}"))
    })?;

    let mut elements = Vec::new();
    let mut imports = Vec::new();

    walk_ast(
        tree.root_node(),
        &rel_path,
        &language,
        &source,
        None,
        &mut elements,
        &mut imports,
    );

    Ok(WalkResult { elements, imports })
}

/// Walk multiple files in batch for better performance.
///
/// Usage from Python:
/// ```python
/// from scholardevclaw_native import walk_batch
/// files = [(source_bytes, rel_path, language), ...]
/// results = walk_batch(files)
/// ```
#[pyfunction]
fn walk_batch(
    _py: Python<'_>,
    files: Vec<(Vec<u8>, String, String)>,
) -> PyResult<Vec<WalkResult>> {
    let mut results = Vec::with_capacity(files.len());

    for (source, rel_path, language) in files {
        let lang = get_language(&language).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Unsupported language: {language}"))
        })?;

        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&lang).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to set language {language}: {e}"
            ))
        })?;

        let tree = parser.parse(&source, None).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to parse {rel_path}"))
        })?;

        let mut elements = Vec::new();
        let mut imports = Vec::new();

        walk_ast(
            tree.root_node(),
            &rel_path,
            &language,
            &source,
            None,
            &mut elements,
            &mut imports,
        );

        results.push(WalkResult { elements, imports });
    }

    Ok(results)
}

/// Check if the native extension is available and working.
#[pyfunction]
fn is_available() -> bool {
    true
}

// ─── Module definition ───────────────────────────────────────────────────────

#[pymodule]
fn scholardevclaw_native(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(walk_file, m)?)?;
    m.add_function(wrap_pyfunction!(walk_batch, m)?)?;
    m.add_function(wrap_pyfunction!(is_available, m)?)?;
    m.add_class::<PyCodeElement>()?;
    m.add_class::<PyImportStatement>()?;
    m.add_class::<WalkResult>()?;
    Ok(())
}
