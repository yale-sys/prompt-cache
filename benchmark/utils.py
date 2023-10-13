from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree
from xml.dom import minidom


class XMLSchemaBuilder:
    def __init__(self, schema_name):
        self.schema = Element('schema', name=schema_name)
        self.user = None
        self.user_union = None

    def set_system_description(self, description):
        system = SubElement(self.schema, 'system')
        system.text = description

    def set_user_description(self, description, user_union=False, scaffold_name="DOC"):
        self.user = SubElement(self.schema, 'user')
        self.user.text = description
        if user_union:
            self.user_union = SubElement(self.user, 'union', scaffold=scaffold_name)

    def add_document_module_union(self, module_name, content):
        assert self.user_union is not None
        module = SubElement(self.user_union, 'module', name=module_name)
        module.text = content.replace('\n', '\\n').replace("'", "\'").replace('"', '\"')

    def add_document_module(self, module_name, content):
        module = SubElement(self.user, 'module', name=module_name)
        module.text = content.replace('\n', '\\n').replace("'", "\'").replace('"', '\"')

    def set_assistant_description(self, description):
        assistant = SubElement(self.schema, 'assistant')
        assistant.text = description

    def generate_xml(self):
        rough_string = tostring(self.schema, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        prettystr = reparsed.toprettyxml(indent="\t")
        return prettystr.replace('&quot;', "'")
