package com.cs512.parser;

import java.io.File;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.DocumentBuilder;
import org.w3c.dom.Document;
import org.w3c.dom.NodeList;
import org.w3c.dom.Node;
import org.w3c.dom.Element;

public class XmlParser {

	public static void main(String[] args) {

		try {
			File inputFile = new File("rss_feed.xml");
			DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
			DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
			Document doc = dBuilder.parse(inputFile);
			doc.getDocumentElement().normalize();
			System.out.println("Root element :" + doc.getDocumentElement().getNodeName());
			NodeList nList = doc.getElementsByTagName("entry");
			System.out.println("----------------------------");
			for (int temp = 0; temp < nList.getLength(); temp++) {
				Node nNode = nList.item(temp);
				System.out.println(" ");
				if (nNode.getNodeType() == Node.ELEMENT_NODE) {
					Element eElement = (Element) nNode;
					String Name = eElement.getElementsByTagName("author").item(0).getTextContent();
					Name = Name.replaceAll("(\\r|\\n|\\t| )", "");
					String Title = eElement.getElementsByTagName("title").item(0).getTextContent();
					String Time = eElement.getElementsByTagName("updated").item(0).getTextContent();
					System.out.println(Name + " : " + Title + " at " + Time);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
