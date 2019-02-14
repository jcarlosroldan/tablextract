function pathTo(element) {
	if (element === document) return ""
	var ix = 0
	var siblings = element.parentNode.childNodes
	for (var i = 0; i < siblings.length; i++) {
		if (siblings[i] === element) return pathTo(element.parentNode) + '/' + element.tagName + '[' + (ix + 1) + ']'
		if (siblings[i].nodeType === 1 && siblings[i].tagName === element.tagName) ix++
	}
}

var removeElements = []
function addRender(subtree) {
	var style = getComputedStyle(subtree)
	if (subtree.tagName == "TR" && subtree.children.length == 0 || subtree.offsetWidth == undefined || style["display"] == "none" || subtree.tagName == "SUP" && subtree.className == "reference") {
		removeElements.push(subtree)
		return
	}
	var serialStyle = ""
	for (let prop of style) {
		if (prop[0] != "-") {
			serialStyle += prop + ":" + style[prop].replace(/:/g, "") + "|"
		}
	}
	serialStyle += "width:" + subtree.offsetWidth / document.body.offsetWidth + "|height:" + subtree.offsetHeight / document.body.offsetHeight
	if (subtree.tagName == "TD" || subtree.tagName == "TH") {
		serialStyle += "|colspan:" + subtree.colSpan + "|rowspan:" + subtree.rowSpan
	}
	subtree.setAttribute("data-computed-style", serialStyle)
	subtree.setAttribute("data-xpath", pathTo(subtree).toLowerCase())
	for (let child of subtree.children) addRender(child)
}

function preprocess() {
	var elements = document.querySelectorAll(injected_script_selector)
	for (let subtree of elements) addRender(subtree)
	for (let elem of removeElements) elem.remove()
}

const injected_script_selector = arguments[0]

if (document.readyState == 'complete') {
	preprocess()
} else {
	window.onload = function(){preprocess()}
}